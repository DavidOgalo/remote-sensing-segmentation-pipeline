import argparse
import glob
import math
import os
import os.path as osp
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Potsdam dataset (standalone version)')
    parser.add_argument('dataset_path', default='data/dataset', help='Path to folder containing 2_Ortho_RGB.zip and 5_Labels_all_noBoundary.zip')
    parser.add_argument('--out_dir', default='data/potsdam', help='Output directory')
    parser.add_argument('--clip_size', type=int, default=512, help='Clipped tile size')
    parser.add_argument('--stride_size', type=int, default=256, help='Stride between tiles')
    args = parser.parse_args()
    return args


def clip_big_image(image_path: str, save_dir: str, args, is_label: bool = False):
    """Crop large TIFF into smaller tiles (and convert labels if needed)."""
    img = Image.open(image_path)
    image = np.array(img)

    h, w = image.shape[:2]
    clip_size = args.clip_size
    stride = args.stride_size

    # Calculate number of tiles
    num_rows = math.ceil((h - clip_size) / stride) + 1 if (h - clip_size) % stride != 0 else math.ceil((h - clip_size) / stride)
    num_cols = math.ceil((w - clip_size) / stride) + 1 if (w - clip_size) % stride != 0 else math.ceil((w - clip_size) / stride)

    # Generate all crop boxes
    for i in range(num_rows):
        for j in range(num_cols):
            start_y = i * stride
            start_x = j * stride
            end_y = min(start_y + clip_size, h)
            end_x = min(start_x + clip_size, w)

            # Only crop if we have at least some overlap
            if end_y - start_y < 50 or end_x - start_x < 50:
                continue

            tile = image[start_y:end_y, start_x:end_x]

            if is_label:
                # Convert RGB label to class index (0-6) exactly like MMSegmentation
                color_map = np.array([
                    [0, 0, 0],      # 0 - impervious surfaces
                    [255, 255, 255],# 1 - building
                    [255, 0, 0],    # 2 - low vegetation
                    [255, 255, 0],  # 3 - tree
                    [0, 255, 0],    # 4 - car
                    [0, 255, 255],  # 5 - clutter
                    [0, 0, 255]     # 6 - background (sometimes used)
                ])
                # Fast vectorized conversion
                flat = tile.reshape(-1, 3)
                distances = np.abs(flat[:, None] - color_map).sum(axis=2)
                class_map = distances.argmin(axis=1).reshape(tile.shape[:2])
                tile = class_map.astype(np.uint8)

            # Filename format: e.g. top_2_10_512_256_1024_768.png
            base_name = osp.basename(image_path).split('.')[0]
            idx_i, idx_j = base_name.split('_')[2:4]
            tile_name = f"{idx_i}_{idx_j}_{start_x}_{start_y}_{end_x}_{end_y}.png"

            Image.fromarray(tile).save(osp.join(save_dir, tile_name))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "img_dir/train").mkdir(parents=True, exist_ok=True)
    (out_dir / "img_dir/val").mkdir(parents=True, exist_ok=True)
    (out_dir / "ann_dir/train").mkdir(parents=True, exist_ok=True)
    (out_dir / "ann_dir/val").mkdir(parents=True, exist_ok=True)

    # Train/val split used by official MMSegmentation
    train_split = {'2_10', '2_11', '2_12', '3_10', '3_11', '3_12', '4_10', '4_11', '4_12',
                   '5_10', '5_11', '5_12', '6_10', '6_11', '6_12', '6_7', '6_8', '6_9',
                   '7_10', '7_11', '7_12', '7_7', '7_8', '7_9'}
    val_split = {'5_15', '6_15', '6_13', '3_13', '4_14', '6_14', '5_14', '2_13',
                 '4_15', '2_14', '5_13', '4_13', '3_14', '7_13'}

    # Find the two zip files
    zips = list(Path(args.dataset_path).glob("*.zip"))
    print(f"Found {len(zips)} zip files: {[z.name for z in zips]}")

    for zipp in zips:
        with tempfile.TemporaryDirectory() as tmp_dir:
            print(f"Extracting {zipp.name} ...")
            with zipfile.ZipFile(zipp) as zf:
                zf.extractall(tmp_dir)

            # Handle nested folder structure sometimes present in the zip
            src_dir = Path(tmp_dir)
            if len(list(src_dir.iterdir())) == 1 and list(src_dir.iterdir())[0].is_dir():
                src_dir = list(src_dir.iterdir())[0]

            tif_files = sorted(glob.glob(str(src_dir / "*.tif")))

            for tif_path in tqdm(tif_files, desc=f"Processing {zipp.name}"):
                patch_id = "_".join(osp.basename(tif_path).split("_")[2:4])

                if "label" in tif_path.lower() or "Labels" in tif_path:
                    data_type = "train" if patch_id in train_split else "val"
                    save_dir = str(out_dir / "ann_dir" / data_type)
                    clip_big_image(tif_path, save_dir, args, is_label=True)
                else:
                    data_type = "train" if patch_id in train_split else "val"
                    save_dir = str(out_dir / "img_dir" / data_type)
                    clip_big_image(tif_path, save_dir, args, is_label=False)

    print("\n Cropping completed!")
    print(f"Output folder: {out_dir}")
    print("   ├── img_dir/train (~3456 images)")
    print("   ├── img_dir/val   (~2016 images)")
    print("   ├── ann_dir/train")
    print("   └── ann_dir/val")


if __name__ == "__main__":
    main()