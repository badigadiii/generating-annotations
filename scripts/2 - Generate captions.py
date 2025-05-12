import os
from PIL import Image
from tqdm import tqdm
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
)
import torch

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

# 🔄 Load model
model_id = "Salesforce/blip-image-captioning-base" if not args.model_id else args.model_id

if "paligemma2" in model_id:
    device = "cuda:0"
    dtype = torch.bfloat16

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=config.PROJECT_PATH / ".cache",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=config.PROJECT_PATH / ".cache")

elif "paligemma" in model_id:
    device = "cuda:0"
    dtype = torch.bfloat16

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        revision="bfloat16",
        cache_dir=config.PROJECT_PATH / ".cache",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=config.PROJECT_PATH / ".cache")
    
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained(model_id, cache_dir=config.PROJECT_PATH / ".cache")
    model = BlipForConditionalGeneration.from_pretrained(model_id, cache_dir=config.PROJECT_PATH / ".cache").to(device)


def generate_caption_default(image_path: str, text: str) -> str:
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption

    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        return ""

def generate_caption_paligemma(image_path: str, prompt: str):
    image = Image.open(image_path).convert("RGB")
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded


generate_caption = None
if "paligemma" in model_id:
    generate_caption = generate_caption_paligemma
else:
    generate_caption = generate_caption_default

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
