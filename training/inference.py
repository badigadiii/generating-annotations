import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_cache", "-m", required=True, help="Path to pretrained model cache")
parser.add_argument("--unet_path", required=True, help="Path to trained unet weights")
parser.add_argument("--dataset_path", required=True)
parser.add_argument("--dataset_config", required=True, default="default", help="Dataset config {default, fold_1, ..., fold_5}")
parser.add_argument("--dataset_split", required=True, help="Dataset split {train, test, validation}")
parser.add_argument("--output_dir", required=True, help="Output dir to store generated images")
parser.add_argument("--image_size", type=int, default=128, required=False)
parser.add_argument("--batch_size", required=True, type=int, help="Batch size of prompts")
parser.add_argument("--num_inference_steps", "-n", type=int, default=2, required=False)
parser.add_argument("--guidance_scale", "-g", type=float, default=7.5, required=False)


args = parser.parse_args()


import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

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

    def generate(self, prompts: list[str], height=256, width=256, num_inference_steps=50, guidance_scale=7.5):
        with torch.autocast(self.device) if self.device == "cuda" else torch.no_grad():
            images = self.pipe(
                prompt=prompts,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images
            return images


model_cache = args.model_cache
unet_path = args.unet_path
dataset_path = args.dataset_path
dataset_config = args.dataset_config
dataset_split = args.dataset_split
output_dir = args.output_dir
image_size = args.image_size

batch_size = args.batch_size
num_inference_steps = args.num_inference_steps
guidance_scale = args.guidance_scale

os.makedirs(output_dir, exist_ok=True)


generator = CustomStableDiffusion(
    model_id="runwayml/stable-diffusion-v1-5",
    cache_dir=model_cache,
    unet_path=unet_path,
)

d = load_dataset(dataset_path, dataset_config)
captions = d[dataset_split]

file_names = captions["file_name"]
prompts = captions["caption"]

for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):

    batch_captions = captions[i:i + batch_size]
    fnames = batch_captions["file_name"]
    prompts = batch_captions["caption"]

    batch_prompts = [] # prompts[i:i + batch_size]
    batch_filenames = [] # file_names[i:i + batch_size]

    for fname, prompt in zip(fnames, prompts):
        img_path = Path(output_dir) / Path(fname)

        if not img_path.exists():
            batch_prompts.append(prompt)
            batch_filenames.append(fname)

    if batch_prompts:
        images = generator.generate(
            prompts=batch_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=image_size,
            width=image_size,
        )

        for img, fname in zip(images, batch_filenames):
            img_path = Path(output_dir) / Path(fname)
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(img_path)