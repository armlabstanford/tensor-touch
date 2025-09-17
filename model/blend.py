import argparse
import os
import os.path as osp
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir1", type=str, required=True)
    parser.add_argument("--input_dir2", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    files1 = os.listdir(args.input_dir1)
    files2 = os.listdir(args.input_dir2)

    os.makedirs(args.output_dir, exist_ok=True)

    for file in tqdm(files1, desc="blend files in 1"):
        os.system(f"ln -s {osp.join(args.input_dir1, file)} {osp.join(args.output_dir, file)}1")
    
    for file in tqdm(files2, desc="blend files in 2"):
        os.system(f"ln -s {osp.join(args.input_dir2, file)} {osp.join(args.output_dir, file)}2")


