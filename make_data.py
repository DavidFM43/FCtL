import os
import shutil
from pathlib import Path
from tqdm import tqdm


def move_data(ids, split, images_dir, masks_dir):
    """Move data to train/val/test folders."""

    # create folders
    os.makedirs(data_dir / split, exist_ok=True)
    split_images_dir = data_dir / split / "Sat"
    split_masks_dir = data_dir / split / "Label"
    os.makedirs(split_images_dir, exist_ok=True)
    os.makedirs(split_masks_dir, exist_ok=True)

    for id in tqdm(ids, desc=f"Preparing {split} data"):
        old_sat = images_dir / f"{id}_sat.jpg"
        old_label = masks_dir / f"{id}_mask.png"
        new_sat = split_images_dir / f"{id}_sat.jpg"
        new_label = split_masks_dir / f"{id}_mask.png"
        shutil.copy(old_label, new_label)
        shutil.copy(old_sat, new_sat)


if __name__ == "__main__":
    data_dir = Path("data")
    raw_data_dir = data_dir / "raw_data"

    images_dir = raw_data_dir / "images"
    masks_dir = raw_data_dir / "masks"

    for split_name in ["train", "crossvali", "test"]:
        with open(data_dir / "splits" / f"{split_name}.txt") as f:
            ids = f.read().splitlines()
        move_data(ids, split_name, images_dir=images_dir, masks_dir=masks_dir)
