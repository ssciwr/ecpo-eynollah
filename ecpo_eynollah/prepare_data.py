# prepare data for training Eynollah
import argparse
from pathlib import Path
import random

SEED = 42
random.seed(SEED)


def prepare_data(
    img_dir: Path,
    labeled_dir: Path,
    train_ratio: float,
    out_train_dir: Path,
    out_eval_dir: Path,
):
    img_dir = img_dir.resolve()
    labeled_dir = labeled_dir.resolve()
    out_train_dir = out_train_dir.resolve()
    out_eval_dir = out_eval_dir.resolve()

    out_train_dir.mkdir(parents=True, exist_ok=True)

    # create images and labels for train folder
    img_train_dir = out_train_dir / "images"
    img_train_dir.mkdir(parents=True, exist_ok=True)
    label_train_dir = out_train_dir / "labels"
    label_train_dir.mkdir(parents=True, exist_ok=True)

    out_eval_dir.mkdir(parents=True, exist_ok=True)
    # create images and labels for eval folder
    img_eval_dir = out_eval_dir / "images"
    img_eval_dir.mkdir(parents=True, exist_ok=True)
    label_eval_dir = out_eval_dir / "labels"
    label_eval_dir.mkdir(parents=True, exist_ok=True)

    labeled_files = list(labeled_dir.glob("*.png"))
    total_files = len(labeled_files)
    train_count = int(total_files * train_ratio)

    # shuffle the files
    random.shuffle(labeled_files)

    for idx, label_file in enumerate(labeled_files):
        img_file = img_dir / label_file.name
        if not img_file.exists():
            print(f"Image file {img_file} does not exist. Skipping.")
            continue

        if idx < train_count:
            # copy to train folder
            dest_img_file = img_train_dir / img_file.name
            dest_label_file = label_train_dir / label_file.name
        else:
            # copy to eval folder
            dest_img_file = img_eval_dir / img_file.name
            dest_label_file = label_eval_dir / label_file.name

        # copy files
        with open(img_file, "rb") as f_src, open(dest_img_file, "wb") as f_dest:
            f_dest.write(f_src.read())
        with open(label_file, "rb") as f_src, open(dest_label_file, "wb") as f_dest:
            f_dest.write(f_src.read())

    print(f"Data preparation completed.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Prepare data for training Eynollah."
    )
    argparser.add_argument(
        "--img_dir",
        type=Path,
        required=True,
        help="Directory containing the downloaded images.",
    )
    argparser.add_argument(
        "--labeled_dir",
        type=Path,
        required=True,
        help="Directory containing the labeled data.",
    )
    argparser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data to total data.",
    )
    argparser.add_argument(
        "--out_train_dir",
        type=Path,
        required=True,
        help="Directory to save the train data.",
    )
    argparser.add_argument(
        "--out_eval_dir",
        type=Path,
        required=True,
        help="Directory to save the evaluation data.",
    )
    args = argparser.parse_args()
    prepare_data(
        args.img_dir,
        args.labeled_dir,
        args.train_ratio,
        args.out_train_dir,
        args.out_eval_dir,
    )
