from dotenv import dotenv_values
from pathlib import Path


class Config:
    env_values = dotenv_values()

    PROJECT_PATH = Path(env_values["PROJECT_PATH"])
    IMAGES_FOLDER = env_values["IMAGES_FOLDER"]
    DATA_FOLDER = env_values["DATA_FOLDER"]

    IMAGES_PATH = PROJECT_PATH / IMAGES_FOLDER
    DATA_PATH = PROJECT_PATH / DATA_FOLDER

    @classmethod
    def as_dict(cls):
        return {k: v for k, v in cls.__dict__.items() if not k.startswith("__")}


config = Config()

if __name__ == "__main__":
    print(config.PROJECT_PATH)
