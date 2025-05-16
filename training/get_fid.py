from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance

from config_file import config

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--real_dir", required=False)
parser.add_argument("--fake_dir", required=False)


args = parser.parse_args()


class ImageFolderWithoutLabels(Dataset):
    def __init__(self, image_dir, transform=None):
        self.paths = list(Path(image_dir).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        img = (img * 255).clamp(0, 255).byte()
        return img  # Оставляем float32 [0.0, 1.0]

def get_fid(real_dir: Path, fake_dir: Path):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    
    real_dataset = ImageFolderWithoutLabels(real_dir, transform=transform)
    fake_dataset = ImageFolderWithoutLabels(fake_dir, transform=transform)

    real_loader = DataLoader(real_dataset, batch_size=32)
    fake_loader = DataLoader(fake_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=2048).to(device)

    with torch.no_grad():
        for imgs in real_loader:
            fid.update(imgs.to(device), real=True)
        for imgs in fake_loader:
            fid.update(imgs.to(device), real=False)

    return fid.compute()

real_dir = config.IMAGES_PATH / "mnk" / "real-validation" if not args.real_dir else args.real_dir
fake_dir = config.IMAGES_PATH / "mnk" / "256x256" / "sd_trained_unet_epoch_80.pt" if not args.fake_dir else args.fake_dir

orig_fid = get_fid(real_dir, fake_dir)
print(f"📊 FID: {orig_fid.item():.4f}")