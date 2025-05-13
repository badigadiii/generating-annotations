import os
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm

from config_file import config

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=False, help="Folder which stores folders with images")
parser.add_argument("-o", "--output", required=False, help="Folder to store folders with transformed frames")

args = parser.parse_args()

# Input
screenshot_folders = config.IMAGES_PATH / "videos-screenshots" if not args.input else args.input
screenshot_folders = Path(screenshot_folders)
folders = os.listdir(screenshot_folders)
# Output
output_folder = config.IMAGES_PATH / "transformed-frames" / "dataset" if not args.output else args.output
output_folder = Path(output_folder)


size = 512
transform = transforms.Compose([
    transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size),
])

for game_index, game_name in enumerate(folders):
    if len(game_name[game_name.rfind(".") + 1:]) != 0: # Change it later. Check if game_name has .ext
        game_name = game_name[:game_name.rfind(".")] # Get rid of .ext in game_name
    input_folder = screenshot_folders / game_name
    output_dir = output_folder / game_name
    os.makedirs(output_dir, exist_ok=True)

    frames = os.listdir(input_folder)

    print(f"\n[{game_index + 1} / {len(folders)}] Started: {game_name}")

    for frame_index, frame_name in tqdm(enumerate(frames), total=len(frames), desc="Framing"):
        img_path = input_folder / frame_name

        if not os.path.exists(output_dir / frame_name):
            img = Image.open(img_path)
            transformed_img = transform(img)
            transformed_img.save(output_dir / frame_name)

            # print(f"[Game {game_index + 1} / {len(folders)}][Frame {frame_index + 1} / {len(frames)}] Saved {output_dir / frame_name}")
