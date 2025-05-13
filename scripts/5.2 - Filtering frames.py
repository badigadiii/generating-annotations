import os
import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm

from config_file import config

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=False, help="Folder which stores folders with images")
parser.add_argument("-o", "--output", required=False, help="Folder to store folders with filtered frames")
parser.add_argument("-v", "--varience_file", required=False, help="File to save info about laplasian varience for each frame")
parser.add_argument("-t", "--threshold", type=float, required=False, help="Threshold for filtering frame by laplasian varience")

args = parser.parse_args()

# Input
input_folder = config.IMAGES_PATH / "transformed-frames" / "dataset" if not args.input else args.input
output_folder = config.IMAGES_PATH / "retro-games-gameplay-frames-30k-512p" / "dataset" if not args.output else args.output
varience_file = config.DATA_PATH / "varience.json" if not args.varience_file else args.varience_file
threshold = 100 if not args.threshold else args.threshold

input_folder = Path(input_folder)
output_folder = Path(output_folder)
varience_file = Path(varience_file)

def variance_of_laplacian(image: Image):
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def check_exposure(image: Image, dark_threshold=30, bright_threshold=225, cutoff=0.7) -> int:
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256]).flatten()
    total = hist.sum()
    
    dark_pixels = hist[:dark_threshold].sum() / total
    bright_pixels = hist[bright_threshold:].sum() / total

    if dark_pixels > cutoff:
        return -1
    elif bright_pixels > cutoff:
        return 1
    
    return 0


games = os.listdir(input_folder)

# Calculate laplasian varience
laplasian_var = {}


if not os.path.exists(varience_file):
    for game_index, game in enumerate(games):
        print(f"[Game {game_index + 1} / {len(games)}] {game}")
        
        frames = os.listdir(input_folder / game)
        laplasian_var[game] = {}

        for frame_index, frame in tqdm(enumerate(frames), total=len(frames), desc="Calculating laplasian"):
            img = Image.open(input_folder / game / frame)
            var = variance_of_laplacian(img)
            laplasian_var[game][frame] = float(var)

    with open(varience_file, "w") as f:
        json.dump(laplasian_var, f, indent=4)


# Filtering
with open(varience_file, "r", encoding="utf-8") as f:
    laplasian_var = json.load(f)

for game_index, game in enumerate(games):
    print(f"[Game {game_index + 1} / {len(games)}] {game}")

    frames = os.listdir(input_folder / game)
    os.makedirs(output_folder / game, exist_ok=True)

    for frame_index, frame in tqdm(enumerate(frames), total=len(frames), desc="Filtering"):
        if not os.path.exists(output_folder / game / frame):
            img = Image.open(input_folder / game / frame)
            if laplasian_var[game][frame] >= threshold and check_exposure(img) == 0:
                img.save(output_folder / game / frame)

# Show info result
before_filtering = 0
for game_index, game in enumerate(os.listdir(input_folder)):
    frames = os.listdir(input_folder / game)
    before_filtering += len(frames)

after_filtering = 0
for game_index, game in enumerate(os.listdir(output_folder)):
    after_filtering += len(os.listdir(output_folder / game))


print("\nComplete!")
print(f"Before filtering: {before_filtering} frames")
print(f"After filtering: {after_filtering} frames")
print(f"{before_filtering - after_filtering} filtered frames")
