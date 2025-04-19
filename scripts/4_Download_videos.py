import pandas as pd
import subprocess
import os
from config_file import config


csv_path = config.DATA_PATH / "videos_small.csv"
df = pd.read_csv(csv_path)

def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip()

df['game'] = df['game'].apply(sanitize_filename)
downloaded = list(map(lambda name: name[:name.rfind('.')].strip(), os.listdir(config.VIDEOS_FOLDER)))
videos_to_download = df[df['game'].apply(lambda game: game not in downloaded)]

for _, row in videos_to_download.iterrows():
    game = row["game"]
    url = row["url"]
    
    print(f"📥 Скачиваем: {game} — {url}")

    # command = [
    #     "yt-dlp",
    #     "-f", "bestvideo[height=1080][tbr>=5000][tbr<=10000] / bestvideo[height=1080]",
    #     "-o", config.VIDEOS_FOLDER / f"{game}.%(ext)s",
    #     "--downloader", "ffmpeg",
    #     "--downloader-args", "ffmpeg:-ss 00:10:00 -t 3600 -r 30",
    #     url
    # ]
    command = [
        "yt-dlp",
        "-f", "bestvideo[height=1080][tbr>=5000][tbr<=10000] / bestvideo[height=1080]",
        "-o", config.VIDEOS_FOLDER / f"{game}.%(ext)s",
        "--downloader", "ffmpeg",
        "--downloader-args", "ffmpeg:-ss 00:10:00 -t 3600",
        url
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"✅ Готово: {game}\n")
    except subprocess.CalledProcessError:
        print(f"❌ Ошибка при скачивании: {game}\n")
