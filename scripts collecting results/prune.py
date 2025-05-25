import pandas as pd
from pathlib import Path
import os
import glob

dataset_path = Path("../datasets/retro-games-gameplay-frames")
df = pd.read_csv(dataset_path / 'train.csv')

fnames = df["file_name"].to_list()
fnames = list(map(lambda p: dataset_path / p, fnames))
fnames[:5]

def convert_to_path(path: str):
    return Path(path)

total_frames = glob.glob(str(dataset_path / "train/*/*.png"))
total_frames = list(map(convert_to_path, total_frames))

frames_to_delete = list(set(total_frames) - set(fnames))

print(len(total_frames), len(fnames))
print(len(frames_to_delete))

## WARNING
# for frame in frames_to_delete:
#     os.remove(str(frame))