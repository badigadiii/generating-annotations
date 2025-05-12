import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDIMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

# TODO: Add argparse

# --------------- Settings ---------------
image_dir = Path("../images") / "core-games"
data_path = Path("../data") / "core_annotations.csv"
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

batch_size = 1
learning_rate = 5e-6
num_epochs = 5
checkpoint_frequency = 2

log_file = Path("training_loss_log.csv")

# --------------- Dataloader ---------------
class CustomDataset(Dataset):
    def __init__(self, image_dir: Path, data_path: Path, transform=None):
        """
        image_dir - директория с изображениями
        data_path - путь к файлу с filename-annotations
        transform - преобразования для изображений
        """
        self.image_dir = image_dir
        self.data = pd.read_csv(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        screenshot = self.data.iloc[idx]
        filepath = self.image_dir / screenshot["filename"]  # 'filename'
        prompt = screenshot["annotation"]

        image = Image.open(filepath).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "prompt": prompt}

size = 512
transform = transforms.Compose([
    transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = CustomDataset(image_dir=image_dir, data_path=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --------------- Training ---------------

# Подгрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "runwayml/stable-diffusion-v1-5"
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
# unet.load_state_dict(torch.load("../unet_final.pt", map_location=device))
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Перемещение моделей на GPU, если доступно
unet.to(device)
vae.to(device)
text_encoder.to(device)

optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

# Процесс обучения
loss_log = []

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
    loss_log.append({"epoch": epoch, "avg_loss": avg_loss})
    print(f"Epoch {epoch}/{num_epochs} | Average Loss: {avg_loss:.6f}")

    # Сохранение чекпоинта по частоте
    if epoch % checkpoint_frequency == 0:
        # TODO: Также сделать попеременное сохранения логов ошибок
        checkpoint_path = checkpoint_dir / f"unet_epoch_{epoch}.pt"
        torch.save(unet.state_dict(), checkpoint_path)
        print(f"Чекпоинт сохранен: {checkpoint_path}")

# Сохранение финальной модели
final_model_path = checkpoint_dir / "unet_final.pt"
torch.save(unet.state_dict(), final_model_path)
print(f"Финальная модель сохранена: {final_model_path}")

# Сохранение логов в CSV для последующего анализа
df_log = pd.DataFrame(loss_log)
df_log.to_csv(log_file, index=False)
print(f"Логи обучения сохранены в: {log_file}")