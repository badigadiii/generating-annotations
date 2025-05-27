import pandas as pd
import subprocess
import os
import logging
import datetime
from config_file import config
from pathlib import Path

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=False, help=".csv file with videos to download")
parser.add_argument("-o", "--output", required=False, help="Folder to store downloaded videos")
parser.add_argument("-d", "--duration", required=False, default="3600", help="Duration off video to download in seconds")
parser.add_argument("-s", "--start", required=False, default="00:10:00", help="Start video to download in HH:MM:SS format")

args = parser.parse_args()

date = datetime.datetime.now()
string_date = date.strftime("%y-%m-%d")

log_file = config.LOGS_PATH / f"{string_date} download.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

input_file = config.DATA_PATH / "videos.csv" if not args.input else args.input
output_folder = config.VIDEOS_FOLDER if not args.output else args.output
output_folder = Path(output_folder)
output_folder.mkdir(parents=True, exist_ok=True)
start = "00:10:00" if not args.start else args.start
duration = "3600" if not args.duration else args.duration

def sanitize_filename(name):
    special = "\\/:*?\"<>|+"
    new_name = "".join(c if c not in special else "_" for c in name)

    if new_name[-1] in " .":
        new_name = new_name[:-1] + "_"

    return new_name


df = pd.read_csv(input_file)
df['game'] = df['game'].apply(sanitize_filename)
downloaded = list(map(lambda name: name[:name.rfind('.')].strip(), os.listdir(output_folder)))
videos_to_download = df[df['game'].apply(lambda game: game not in downloaded)]

for i, row in videos_to_download.iterrows():
    game = row["game"]
    url = row["url"]
    
    print(f"📥 Скачиваем [{i+1}/{len(df)}] {game} — {url}")
    logging.info(f"📥 Скачиваем [{i+1}/{len(df)}] {game} — {url}")

    command = [
        "yt-dlp",
        "-f", "bestvideo[height=1080][tbr>=5000][tbr<=10000] / bestvideo[height=1080]",
        "-o", str(output_folder / f"{game}.%(ext)s"),
        "--downloader", "ffmpeg",
        "--downloader-args", f"ffmpeg:-ss {start} -t {duration}",
        url
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Готово [{i+1}/{len(df)}]: {game}")
        logging.info(f"✅ Готово [{i+1}/{len(df)}]: {game}")
    except subprocess.CalledProcessError:
        print(f"❌ Ошибка при скачивании: {game}")
        logging.info(f"❌ Ошибка при скачивании: {game}")
