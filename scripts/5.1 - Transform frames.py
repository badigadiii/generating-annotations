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
input_folders = config.IMAGES_PATH / "videos-screenshots" if not args.input else args.input
input_folders = Path(input_folders)
folders = os.listdir(input_folders)
# Output
output_folder = config.IMAGES_PATH / "transformed-frames" / "dataset" if not args.output else args.output
output_folder = Path(output_folder)


size = 512
transform = transforms.Compose([
    transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size),
])

for game_index, game_name in enumerate(folders):
    game_folder = input_folders / game_name

    new_game_name = game_name
    if len(new_game_name[new_game_name.rfind(".") + 1:]) != 0: # Change it later. Check if game_name has .ext
        new_game_name = new_game_name[:new_game_name.rfind(".")] # Get rid of .ext in game_name
    
    save_dir = output_folder / new_game_name
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[{game_index + 1} / {len(folders)}] Started: {game_name}")

    frames = os.listdir(game_folder)
    for frame_index, frame_name in tqdm(enumerate(frames), total=len(frames), desc="Transforming frames"):
        img_path = game_folder / frame_name

        if not os.path.exists(save_dir / frame_name):
            img = Image.open(img_path)
            transformed_img = transform(img)
            transformed_img.save(save_dir / frame_name)

            # print(f"[Game {game_index + 1} / {len(folders)}][Frame {frame_index + 1} / {len(frames)}] Saved {output_dir / frame_name}")
