import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fake_dir", required=False)
parser.add_argument("--real_dir", required=False)
parser.add_argument("--output", "-o", required=False)

args = parser.parse_args()


from cleanfid import fid
from config_file import config


fake_dir =  args.fake_dir # config.PROJECT_PATH / config._DATASETS_FOLDER / "flat" / "fake_batch-16_lr_5e-6_fold_1_128px"
real_dir =  args.real_dir # config.PROJECT_PATH / config._DATASETS_FOLDER / "flat" / "fold_1"
output =  args.output # "result.txt"

score = fid.compute_fid(str(fake_dir), str(real_dir), mode="clean", num_workers=0)

print("FID:", score)

with open(output, "w") as f:
    f.write(str(score))
