import os
from PIL import Image
from pathlib import Path

from config_file import config

class RetroGames:
    def __init__(self, path_to_folder: str | Path, path_to_captions: str | Path | None = None):
        self.path_to_folder: Path = Path(path_to_folder)
        self.path_to_captions: Path | None = path_to_captions
        
        if path_to_captions:
            self.path_to_captions = Path(path_to_captions)
            self.captions: list[str] = self._parse_captions()
    
    def get_games(self) -> list[str]:
        games = []
        for d in self.path_to_folder.iterdir():
            if d.is_dir():
                games.append(d.parts[-1])

        return games
    
    def get_frames(self, game: str) -> list[Path]:
        return list(self.path_to_folder.glob(f"**/{game}/*.png"))
    
    def get_image_frame(self, frame: str | Path):
        img = Image.open(frame)
        return img
    
    def _parse_captions(self) -> dict[str | Path, str]:
        if not self.path_to_captions.exists():
            raise ValueError(f"Captions file {self.path_to_captions} doesn't exists")
        
        captions = {}
        
        with open(self.path_to_captions, "r", encoding="utf-8") as f:
            for line in f:
                frame_path, caption = line.split("\t")
                frame_path = Path(frame_path)
                game_name = frame_path.parts[-2]

                if game_name not in captions:
                    captions[game_name] = {}
                captions[game_name][frame_path] = caption.strip()
        
        return captions
    
    def get_captions(self):
        if not self.path_to_captions.exists():
            raise ValueError(f"No captions file")
        
        return self.captions

    def get_game_captions(self, game: str) -> dict[str | Path, str]:
        if not self.path_to_captions.exists():
            raise ValueError(f"No captions file")
        
        return self.captions[game]
    
    def get_caption(self, frame: str | Path) -> str:
        if not self.path_to_captions.exists():
            raise ValueError(f"No captions file")

        game = Path(frame).parts[-2]
        return self.captions[game][frame]
    
    def get_all_frames(self):
        for game in self.get_games():
            for frame in self.get_frames(game):
                yield frame


if __name__ == "__main__":
    dataset_path = config.DATASET_PATH
    captions_path = dataset_path / "test.csv"

    r = RetroGames(dataset_path / "test", captions_path)
    games = r.get_games()
    print(games)
    frames = r.get_frames(games[0])