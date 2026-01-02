# prepare data for training Eynollah
import argparse
from pathlib import Path
import random
import click

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

    # group images by year and page type
    # year: characters 9th to 12th in the filename
    # page type: *0001*0004.png, or *0002*0003.png, *0005*0008.png, *0006*0007.png, etc.
    # within each group, shuffle and split according to train_ratio
    groups = {}
    for label_file in labeled_files:
        filename = label_file.stem  # without extension
        year = filename[8:12]
        page_num = filename.split("_")[-1]  # get the page number part
        first_page = int(page_num[:4])
        second_page = int(page_num[-4:])
        page_type = f"{int(first_page)}-{int(second_page)}"
        group_key = (year, page_type)
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(label_file)

    # for debugging
    print(f"Total labeled files: {total_files}")
    print(f"Total groups: {len(groups)}")
    print("Group sizes:")
    for group_key, files in groups.items():
        print(f"{group_key}: {len(files)}")

        # split and copy files

        random.shuffle(files)
        group_size = len(files)
        group_train_count = int(group_size * train_ratio)

        for idx, label_file in enumerate(files):
            img_file = img_dir / label_file.name
            if not img_file.exists():
                print(f"Image file {img_file} does not exist. Skipping.")
                continue

            if idx < group_train_count:
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


@click.option(
    "--img_dir",
    type=click.Path(
        writable=False,
        file_okay=False,
        dir_okay=True,
        exists=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Directory containing the downloaded images.",
)
@click.option(
    "--labeled_dir",
    type=click.Path(
        writable=False,
        file_okay=False,
        dir_okay=True,
        exists=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Directory containing the labeled data.",
)
@click.option(
    "--train_ratio",
    type=float,
    default=0.8,
    help="Ratio of training data to total data.",
)
@click.option(
    "--out_train_dir",
    type=click.Path(
        writable=True,
        file_okay=False,
        dir_okay=True,
        exists=False,
        resolve_path=True,
        path_type=Path,
    ),
    help="Directory to save the train data.",
)
@click.option(
    "--out_eval_dir",
    type=click.Path(
        writable=True,
        file_okay=False,
        dir_okay=True,
        exists=False,
        resolve_path=True,
        path_type=Path,
    ),
    help="Directory to save the evaluation data.",
)
@click.command
def prepare_data_cli(img_dir, labeled_dir, train_ratio, out_train_dir, out_eval_dir):
    prepare_data(
        img_dir,
        labeled_dir,
        train_ratio,
        out_train_dir,
        out_eval_dir,
    )


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
