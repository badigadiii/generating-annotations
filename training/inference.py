import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from pathlib import Path

from config_file import config


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


checkpoints_dir = Path("./checkpoints")

generator = CustomStableDiffusion(
    model_id="runwayml/stable-diffusion-v1-5",
    unet_path=checkpoints_dir / "unet_epoch_80.pt"
)

img = generator.generate("Girl with big sword in armor", num_inference_steps=5)
img.save("image.png")