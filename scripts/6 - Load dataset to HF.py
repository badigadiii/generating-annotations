from config_file import config
from huggingface_hub import HfApi

api = HfApi()

dataset_folder = config.DATA_PATH / 'retro-games-gameplay-frames-30k-512p'

# huggingface-cli upload-large-folder badigadiii/retro-games-gameplay-frames --repo-type dataset retro-games-gameplay-frames --num-workers=4
api.upload_large_folder(
    repo_id="badigadiii/retro-games-gameplay-frames-30k-512p",
    repo_type="dataset",
    folder_path=dataset_folder,
    num_workers=4
)