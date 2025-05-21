import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fake_dir", required=True)
parser.add_argument("--real_dir", required=True)
parser.add_argument("--output", required=True)

args = parser.parse_args()


from cleanfid import fid

fake_dir = args.fake_dir
real_dir = args.real_dir
output = args.output

score = fid.compute_fid(str(fake_dir), str(real_dir), mode="clean", num_workers=0)

print("FID:", score)

with open(output, "w") as f:
    f.write(str(score))
