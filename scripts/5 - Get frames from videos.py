import logging
import subprocess
from pathlib import Path
import os
from config_file import config

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

frames_folder = config.IMAGES_PATH / "videos-screenshots"

videos = os.listdir(config.VIDEOS_FOLDER)
existing_frame_folders = os.listdir(frames_folder)

videos_to_frames = list(set(videos).difference(set(existing_frame_folders)))


# ffmpeg -i "F:\Videos\Hitman_ Blood Money.mp4" -vf "select='gt(scene,0.1)',showinfo" -vsync vfr "images/videos-screenshots/Hitman_ Blood Money/frame_%04d.png"

for i, video in enumerate(videos_to_frames):
    print(f"📥 Фреймирование: {video}")
    logging.info(f"📥 Фреймирование: {video}")

    os.makedirs(frames_folder / video, exist_ok=True)

    command = [
        "ffmpeg",
        "-i", f"{config.VIDEOS_FOLDER}/{video}",
        "-vf", "select='gt(scene,0.2)',showinfo",
        "-fps_mode", "vfr",
        f"{frames_folder}/{video}/frame_%04d.png"
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Готово [{(i + 1) + len(existing_frame_folders)}/{len(videos)}]: {video}\n")
        logging.info(f"✅ Готово [{(i + 1) + len(existing_frame_folders)}/{len(videos)}]: {video}")
    except subprocess.CalledProcessError:
        print(f"❌ Ошибка при фреймировании: {video}\n")
        logging.info(f"❌ Ошибка при фреймировании: {video}\n")