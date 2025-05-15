import os
from PIL import Image
from pathlib import Path
import pandas as pd
import random

from config_file import config

class RetroGamesHelper:
    def __init__(self, path_to_folder: str | Path, path_to_captions: str | Path | None = None):
        self.path_to_folder: Path = Path(path_to_folder)
        self.path_to_captions: Path | None = path_to_captions
        
        if path_to_captions:
            self.path_to_captions = Path(path_to_captions)
            self.captions = pd.read_csv(self.path_to_captions)
    
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
    
    def get_captions(self):
        if not self.path_to_captions.exists():
            raise ValueError(f"No captions file")
        
        return self.captions

    def get_game_captions(self, game_or_games: str | list[str]) -> dict[str | Path, str]:
        if not self.path_to_captions.exists():
            raise ValueError(f"No captions file")
        
        def is_game(filename):
            game_from_path = Path(filename).parts[-2]

            if isinstance(game_or_games, list):
                return game_from_path in game_or_games
            return game_from_path == game_or_games
        
        is_game_captions = self.captions["file_name"].apply(is_game)

        return self.captions[is_game_captions]
    
    def get_caption(self, frame: str | Path) -> str:
        if not self.path_to_captions.exists():
            raise ValueError(f"No captions file")

        return self.captions[self.captions['file_name'] == str(frame)]
    
    def get_all_frames(self):
        for game in self.get_games():
            for frame in self.get_frames(game):
                yield frame
    
    def get_fold(self, fold_index: int, k_folds: int) -> tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
        if not (0 <= fold_index < k_folds):
            raise ValueError(f"Wrong fold index: {fold_index}; 0 <= fold_index < {k_folds}")
        
        random.seed(1234)
        games = self.get_games()
        random.shuffle(games)

        val_fold_size = len(games) // k_folds

        for i in range(0, len(games), val_fold_size):
            val_fold = games[i: i + val_fold_size]

            if len(val_fold) < val_fold_size:
                game_bunch = list(set(games) - set(val_fold))
                random.shuffle(game_bunch)
                val_fold += game_bunch[:val_fold_size - len(val_fold)]

            train_fold = list(set(games) - set(val_fold))

            train_fold_captions = self.get_game_captions(train_fold)
            val_fold_captions = self.get_game_captions(val_fold)

            if i // val_fold_size == fold_index:
                return train_fold_captions, val_fold_captions


if __name__ == "__main__":
    dataset_path = config.DATASET_PATH
    captions_path = dataset_path / "test.csv"

    r = RetroGamesHelper(dataset_path / "test", captions_path)
    games = r.get_games()
    frames = r.get_frames(games[0])
    captions = r.get_game_captions(games[:2])
    print(len(captions))
    # frame = Path(r"test\Bayonetta\frame_0010.png")
    # print(r.get_caption(frame))