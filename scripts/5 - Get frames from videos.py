import logging
import subprocess
from pathlib import Path
import os
from config_file import config

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=False, help="Folder with videos to extract frames")
parser.add_argument("-o", "--output", required=False, help="Folder to store videos frames")
parser.add_argument("-f", "--frame_rate", required=False, type=float)
parser.add_argument("-s", "--scene_difference", required=False, help="From 0 ot 1 number", type=float)


args = parser.parse_args()


log_file = config.LOGS_PATH / "framing_videos.log"
log_file.parent.mkdir(parents=True, exist_ok=True)  # На всякий случай создаём директорию
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

input_folder = config.VIDEOS_FOLDER if not args.input else args.input
input_folder = Path(input_folder)
output_folder = config.IMAGES_PATH / "videos-screenshots" if not args.output else args.output
output_folder = Path(output_folder)
os.makedirs(output_folder, exist_ok=True)

scene_difference = 0.2 if not args.scene_difference else args.scene_difference
frame_rate = 1 if not args.frame_rate else args.frame_rate


videos = os.listdir(input_folder)
framed_videos = os.listdir(output_folder)
# videos_to_frames = list(set(videos).difference(set(framed_videos)))


# ffmpeg -i "F:\Videos\Hitman_ Blood Money.mp4" -vf "select='gt(scene,0.1)',showinfo" -vsync vfr "images/videos-screenshots/Hitman_ Blood Money/frame_%04d.png"

for i, video in enumerate(videos):
    print(f"📥 Фреймирование: {video}")
    logging.info(f"📥 Фреймирование: {video}")

    if os.path.exists(output_folder / video):
        print(f"✅ Уже Готово [{(i + 1)}/{len(videos)}]: {video}\n")
        logging.info(f"✅ Уже Готово [{(i + 1)}/{len(videos)}]: {video}")
        continue

    os.makedirs(output_folder / video, exist_ok=True)

    command = [
        "ffmpeg",
        "-i", str(input_folder / video),
        "-vf", f"fps={frame_rate},blackframe=90:32,mpdecimate,select='gt(scene,{scene_difference})",
        "-fps_mode", "vfr",
        f"{output_folder}/{video}/frame_%04d.png"
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Готово [{(i + 1)}/{len(videos)}]: {video}\n")
        logging.info(f"✅ Готово [{(i + 1)}/{len(videos)}]: {video}")
    except subprocess.CalledProcessError:
        print(f"❌ Ошибка при фреймировании: {video}\n")
        logging.info(f"❌ Ошибка при фреймировании: {video}\n")