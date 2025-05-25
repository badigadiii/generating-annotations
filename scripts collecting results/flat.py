import os
from pathlib import Path
import shutil

def flatten_dataset(src_root, dst_root, exts={'.png', '.jpg', '.jpeg'}):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    count = 0
    for image_path in src_root.rglob("*"):
        if image_path.suffix.lower() not in exts:
            continue

        # Название с уникальностью: game1_frame00123.png
        parts = image_path.parts[-2:]  # например: ['game1', 'frame123.png']
        new_name = "#".join(parts)
        new_path = dst_root / new_name

        shutil.copy(image_path, new_path)
        count += 1

    print(f"[✓] Скопировано {count} файлов в {dst_root}")

# Пример использования:
if __name__ == "__main__":
    flatten_dataset(
        "../datasets/retro-games-gameplay-frames/train",
        "../datasets/retro-flat"
    )
