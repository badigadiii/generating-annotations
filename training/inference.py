import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from pathlib import Path

from config_file import config
from dataset import RetroGamesHelper


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--unet_path", required=True)
parser.add_argument("--images_dir", required=True)
parser.add_argument("--dataset_path", required=True)
parser.add_argument("--cache_dir", required=True)
parser.add_argument("--num_inference_steps", "-n", type=int, default=2, required=False)
parser.add_argument("--guidance_scale", "-g", type=float, default=7.5, required=False)


args = parser.parse_args()


class CustomStableDiffusion:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", unet_path=None, device=None, torch_dtype=torch.float16, cache_dir="./models"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", cache_dir=cache_dir),
            safety_checker=None,
            cache_dir=cache_dir
        ).to(self.device)

        # Заменяем UNet на кастомный (если указан путь)
        if unet_path:
            print(f"🔁 Loading custom UNet from: {unet_path}")
            unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", cache_dir=cache_dir)
            state_dict = torch.load(unet_path, map_location=self.device)
            unet.load_state_dict(state_dict)
            unet.to(self.device)
            self.pipe.unet = unet

    def generate(self, prompt, height=256, width=256, num_inference_steps=50, guidance_scale=7.5):
        with torch.autocast(self.device) if self.device == "cuda" else torch.no_grad():
            image = self.pipe(
                prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            return image


unet_path = Path(args.unet_path)
images_dir = Path(args.images_dir)
dataset_path = Path(args.dataset_path)

os.makedirs(images_dir, exist_ok=True)

retro_helper = RetroGamesHelper(dataset_path / "test", dataset_path / "test.csv")
captions = retro_helper.get_captions()
for i in range(len(captions)):
    item = captions.iloc[i]
    filename = Path(item['file_name'])
    caption = item['caption']
    os.makedirs(images_dir / filename.parent, exist_ok=True)

    generator = CustomStableDiffusion(
        model_id="runwayml/stable-diffusion-v1-5",
        unet_path=unet_path,
        cache_dir=args.cache_dir
    )

    img = generator.generate(caption, num_inference_steps=args.num_inference_steps)
    img.save(images_dir / filename)
    break