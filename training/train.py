# ---------- Args ----------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", "-data", required=False, help="Dataset directory")
parser.add_argument("--dataset_split", "-split", required=False, help="Split from dataset {train, test}")
parser.add_argument("--dataset_config", required=False, help="Dataset config: default, fold_1, ..., fold_5")
parser.add_argument("--checkpoints_dir", "-c", required=False)
parser.add_argument("--log_file", "-log", required=False, help="Path to training log file")
parser.add_argument("--unet_weights", "-w", type=int, required=False, help="File with unet weights")
parser.add_argument("--model_cache", "-m", required=False, help="Path to pretrained model cache")
parser.add_argument("--batch_size", "-b", type=int, required=False)
parser.add_argument("--num_epochs", "-n", type=int, required=False)
parser.add_argument("--checkpoint_frequency", "-freq", type=int, required=False)
parser.add_argument("--learning_rate", "-lr", type=float, required=False)


args = parser.parse_args()


import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDIMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset

from dataset import RetroGamesHelper
from config_file import config

# ---------- Logging ----------
import logging

log_file = config.LOGS_PATH / "training_log.log" if not args.log_file else args.log_file
log_file = Path(log_file)
os.makedirs(log_file.parent, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Логгер инициализирован.")
logger.info(args)


# --------------- Settings ---------------
dataset_path = Path("./dataset") if not args.dataset_path else args.dataset_path
dataset_split = "train" if not args.dataset_split else args.dataset_split
dataset_config = "default" if not args.dataset_config else args.dataset_config
checkpoints_dir = config.CHECKPOINTS_PATH if not args.checkpoints_dir else args.checkpoints_dir
cache_dir = "./models" if not args.model_cache else args.model_cache

dataset_path = Path(dataset_path)
checkpoints_dir = Path(checkpoints_dir)
checkpoints_dir.mkdir(exist_ok=True, parents=True)

batch_size = 1 if not args.batch_size else args.batch_size
learning_rate = 5e-6 if not args.learning_rate else args.learning_rate
num_epochs = 5 if not args.num_epochs else args.num_epochs
checkpoint_frequency = 2 if not args.checkpoint_frequency else args.checkpoint_frequency

# --------------- Dataloader ---------------
class CustomDataset(Dataset):
    def __init__(self, dataset_path: Path, dataset_split: str, dataset_config: str, transform=None):
        self.dataset_path = Path(dataset_path)
        dataset = load_dataset(str(dataset_path), dataset_config)
        self.dataset = dataset[dataset_split]

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        screenshot = self.dataset[idx]
        filepath = self.dataset_path / screenshot["file_name"]
        prompt = screenshot["caption"]

        image = Image.open(filepath).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "prompt": prompt}

size = 128
transform = transforms.Compose([
    transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = CustomDataset(dataset_path=dataset_path, dataset_split=dataset_split, dataset_config=dataset_config, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --------------- Training ---------------

# Подгрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "runwayml/stable-diffusion-v1-5"
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", cache_dir=cache_dir)
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", cache_dir=cache_dir)
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", cache_dir=cache_dir)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", cache_dir=cache_dir)

if args.unet_weights:
    unet.load_state_dict(torch.load(args.unet_weights, map_location=device))

scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Перемещение моделей на GPU, если доступно
unet.to(device)
vae.to(device)
text_encoder.to(device)

optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

# Процесс обучения
for epoch in range(1, num_epochs + 1):
    epoch_losses = []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)

    for step, batch in enumerate(progress_bar):
        optimizer.zero_grad()

        pixel_values = batch["pixel_values"].to(device)
        batch_prompt = batch["prompt"]

        # Токенизация текста
        text_inputs = tokenizer(
            batch_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        text_embeddings = text_encoder(text_input_ids)[0]

        # Кодирование изображений в латентное пространство с помощью VAE
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * 0.18215

        # Добавление случайного шума
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
        ).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        noise_pred = unet(
            noisy_latents, timesteps, encoder_hidden_states=text_embeddings
        ).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        progress_bar.set_postfix({"Loss": loss.item()})

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Epoch {epoch}/{num_epochs} | Average Loss: {avg_loss:.6f}")
    logger.info(f"Epoch {epoch}/{num_epochs} | Average Loss: {avg_loss:.6f}")

    # Сохранение чекпоинта по частоте
    if epoch % checkpoint_frequency == 0:
        checkpoint_path = checkpoints_dir / f"unet_epoch_{epoch}.pt"
        torch.save(unet.state_dict(), checkpoint_path)
        print(f"Чекпоинт сохранен: {checkpoint_path}")
        logger.info(f"Чекпоинт сохранен: {checkpoint_path}")


# Сохранение финальной модели
final_model_path = checkpoints_dir / "unet_final.pt"
torch.save(unet.state_dict(), final_model_path)
print(f"Финальная модель сохранена: {final_model_path}")
logger.info(f"Финальная модель сохранена: {final_model_path}")


# Сохранение логов для последующего анализа
logger.info(f"Логи обучения сохранены в: {log_file}")
