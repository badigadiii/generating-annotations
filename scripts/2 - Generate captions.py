import os
from PIL import Image
from tqdm import tqdm
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
)
import torch
import pandas as pd
import json

from config_file import config

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", "-m", required=False)
parser.add_argument("--output", "-o", required=False)
parser.add_argument("--prompt", "-p", required=False)
args = parser.parse_args()

# Constants
images_folder = config.IMAGES_PATH / "retro-games-gameplay-frames-30k-512p" / "dataset"

output_file = config.DATA_PATH / "captions.txt" if not args.output else args.output
prompt = "A screenshot from a video game shows" if not args.prompt else args.prompt

# ⚙️ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 🔄 Load model
model_id = "Salesforce/blip-image-captioning-base" if not args.model_id else args.model_id

processor = BlipProcessor.from_pretrained(model_id, cache_dir=config.PROJECT_PATH / ".cache")
model = BlipForConditionalGeneration.from_pretrained(model_id, cache_dir=config.PROJECT_PATH / ".cache").to(device)


def generate_caption(image_path: str, text: str) -> str:
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption

    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        return ""


# Generate captions
generated_captions = []

if os.path.exists(output_file):
    with open(output_file, "r") as f:
        lines = f.readlines()
    generated_captions = list(map(lambda line: line.split('\t')[0], lines))

for game in tqdm(os.listdir(images_folder), desc="Генерация аннотаций"):
    for frame in tqdm(os.listdir(images_folder / game), desc=f"Генерация аннотаций для {game}"):

        if str(images_folder / game / frame) not in generated_captions:
            caption = generate_caption(images_folder / game / frame, prompt)

            with open(output_file, "a") as f:
                f.write(f"{images_folder / game / frame}\t{caption}\n")