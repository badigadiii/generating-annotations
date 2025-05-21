from dotenv import dotenv_values
from pathlib import Path


class Config:
    env_values = dotenv_values()

    PROJECT_PATH = Path(env_values["PROJECT_PATH"])
    _IMAGES_FOLDER = Path(env_values["IMAGES_FOLDER"])
    _DATA_FOLDER = Path(env_values["DATA_FOLDER"])
    _LOGS_FOLDER = Path(env_values["LOGS_FOLDER"])
    _DATASETS_FOLDER = Path(env_values["DATASETS_FOLDER"])
    VIDEOS_FOLDER = Path(env_values["VIDEOS_FOLDER"])
    DATASET_NAME = Path(env_values["DATASET_NAME"])
    
    HUG_API_TOKEN = env_values["HUG_API_TOKEN"]

    IMAGES_PATH = PROJECT_PATH / _IMAGES_FOLDER
    DATA_PATH = PROJECT_PATH / _DATA_FOLDER
    LOGS_PATH = PROJECT_PATH / _LOGS_FOLDER
    DATASET_PATH = PROJECT_PATH / _DATASETS_FOLDER / DATASET_NAME

    @classmethod
    def as_dict(cls):
        return {k: v for k, v in cls.__dict__.items() if not k.startswith("__")}


config = Config()

if __name__ == "__main__":
    print(config.PROJECT_PATH)
