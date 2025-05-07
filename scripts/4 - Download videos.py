import pandas as pd
import subprocess
import os
import logging
from config_file import config

log_file = config.LOGS_PATH / "download.log"
log_file.parent.mkdir(parents=True, exist_ok=True)  # На всякий случай создаём директорию
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

csv_path = config.DATA_PATH / "videos.csv" # "videos_small.csv"
df = pd.read_csv(csv_path)

def sanitize_filename(name):
    special = "\\/:*?\"<>|+"
    new_name = "".join(c if c not in special else "_" for c in name)

    if new_name[-1] in " .":
        new_name = new_name[:-1]

    return new_name

df['game'] = df['game'].apply(sanitize_filename)
downloaded = list(map(lambda name: name[:name.rfind('.')].strip(), os.listdir(config.VIDEOS_FOLDER)))
videos_to_download = df[df['game'].apply(lambda game: game not in downloaded)]

for i, row in videos_to_download.iterrows():
    game = row["game"]
    url = row["url"]
    
    print(f"📥 Скачиваем: {game} — {url}")
    logging.info(f"📥 Скачиваем: {game} — {url}")

    command = [
        "yt-dlp",
        "-f", "bestvideo[height=1080][tbr>=5000][tbr<=10000] / bestvideo[height=1080]",
        "-o", str(config.VIDEOS_FOLDER / f"{game}.%(ext)s"),
        "--downloader", "ffmpeg",
        "--downloader-args", "ffmpeg:-ss 00:10:00 -t 3600",
        url
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Готово [{i+1}/{len(df)}]: {game}\n")
        logging.info(f"✅ Готово [{i+1}/{len(df)}]: {game}")
    except subprocess.CalledProcessError:
        print(f"❌ Ошибка при скачивании: {game}\n")
        logging.info(f"❌ Ошибка при скачивании: {game}\n")
