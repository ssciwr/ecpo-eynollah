# prepare data for training Eynollah
import argparse
from pathlib import Path
import random
import click

SEED = 42
random.seed(SEED)

ads_sep_conditions = {
    "1920": {
        "0001": "ads-heavy",
        "0004": "ads-heavy",
        "0002": "text-heavy",
        "0003": "text-heavy",
    },
    "1930": {
        "0001": "ads-heavy",
        "0004": "ads-heavy",
        "0002": "text-heavy",
        "0003": "text-heavy",
    },
    "1939": {
        "0001": "text-heavy",
        "0004": "text-heavy",
        "0002": "text-heavy",
        "0003": "text-heavy",
        "0005": "text-heavy",
        "0008": "text-heavy",
        "0006": "text-heavy",
        "0007": "text-heavy",
    },
}


def separate_ads_text_heavy_files(labeled_files, ads_sep_conditions):

    ads_heavy_files = set()
    text_heavy_files = set()
    dup_files = []

    for label_file in labeled_files:
        if "_dup" in label_file.stem:
            dup_files.append(label_file)
            continue  # duplicated files will be processed later

        filename = label_file.stem  # without extension
        year = filename[8:12]
        page_num = filename.split("_")[-1]  # get the page number part
        first_page = page_num[:4]
        second_page = page_num[-4:]

        if year in ads_sep_conditions:
            if first_page in ads_sep_conditions[year]:
                condition = ads_sep_conditions[year][first_page]
                if condition == "ads-heavy":
                    ads_heavy_files.add(label_file)
                elif condition == "text-heavy":
                    text_heavy_files.add(label_file)
                else:
                    raise ValueError(
                        f"Unknown condition {condition} for file {label_file}"
                    )
            elif (second_page != first_page) and (
                second_page in ads_sep_conditions[year]
            ):
                condition = ads_sep_conditions[year][second_page]
                if condition == "ads-heavy":
                    ads_heavy_files.add(label_file)
                elif condition == "text-heavy":
                    text_heavy_files.add(label_file)
                else:
                    raise ValueError(
                        f"Unknown condition {condition} for file {label_file}"
                    )
            else:
                raise ValueError(
                    f"No condition found for file {label_file} in year {year}, pages {first_page} and {second_page}"
                )
        else:
            raise ValueError(
                f"No conditions defined for year {year}. File {label_file} will be ignored for separation."
            )

    if dup_files:
        print(
            f"Found {len(dup_files)} duplicated files. Processing them to assign to the correct sets..."
        )
    for dup_file in dup_files:
        parent_folder = dup_file.parent
        original_name = parent_folder / dup_file.name.replace("_dup", "")
        if original_name in ads_heavy_files:
            ads_heavy_files.add(dup_file)
        elif original_name in text_heavy_files:
            text_heavy_files.add(dup_file)
        else:
            print(
                f"Original file for {dup_file} not found in either ads-heavy or text-heavy sets. Skipping this dup file."
            )

    return ads_heavy_files, text_heavy_files


def group_and_split_files(
    labeled_files,
    train_ratio,
    img_dir,
    train_imgs_path,
    train_labels_path,
    eval_imgs_path,
    eval_labels_path,
):
    # group images by year and page type
    # year: characters 9th to 12th in the filename
    # page type: *0001*0004.png, or *0002*0003.png, *0005*0008.png, *0006*0007.png, etc.
    # within each group, shuffle and split according to train_ratio
    # dup files are considered the same as original files
    total_files = len(labeled_files)
    groups = {}
    dup_files = []
    for label_file in labeled_files:
        filename = label_file.stem  # without extension
        if "_dup" in filename:
            dup_files.append(label_file)
            continue

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
    print("Group sizes (without dup files):")

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
                dest_img_file = train_imgs_path / img_file.name
                dest_label_file = train_labels_path / label_file.name
            else:
                # copy to eval folder
                dest_img_file = eval_imgs_path / img_file.name
                dest_label_file = eval_labels_path / label_file.name

            # copy files
            with open(img_file, "rb") as f_src, open(dest_img_file, "wb") as f_dest:
                f_dest.write(f_src.read())
            with (
                open(label_file, "rb") as f_src,
                open(dest_label_file, "wb") as f_dest,
            ):
                f_dest.write(f_src.read())

    # process dup files
    # dup files are added to the same folder as their original files
    print(f"Total dup files to process: {len(dup_files)}")
    for dup_file in dup_files:
        original_name = dup_file.name.replace("_dup", "")
        # determine if original file is in train or eval
        if (train_labels_path / original_name).exists():
            dest_label_file = train_labels_path / dup_file.name
            dest_img_file = train_imgs_path / dup_file.name
        elif (eval_labels_path / original_name).exists():
            dest_label_file = eval_labels_path / dup_file.name
            dest_img_file = eval_imgs_path / dup_file.name
        else:
            print(
                f"Original file for {dup_file} not found in train or eval directories. Skipping."
            )
            continue

        # copy dup files
        with (
            open(img_dir / dup_file.name, "rb") as f_src,
            open(dest_img_file, "wb") as f_dest,
        ):
            f_dest.write(f_src.read())
        with open(dup_file, "rb") as f_src, open(dest_label_file, "wb") as f_dest:
            f_dest.write(f_src.read())


def prepare_data(
    img_dir: Path,
    labeled_dir: Path,
    train_ratio: float,
    ads_separation: bool,
    out_dir: Path,
):
    img_dir = img_dir.resolve()
    labeled_dir = labeled_dir.resolve()
    out_dir = out_dir.resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    parent_folders = ["ads", "text"] if ads_separation else ["mixed"]

    # for each parent folder, create train and eval folders
    # in each train and eval folder, create images and labels folders
    for parent in parent_folders:
        (out_dir / parent).mkdir(parents=True, exist_ok=True)
        (out_dir / parent / "train").mkdir(parents=True, exist_ok=True)
        (out_dir / parent / "eval").mkdir(parents=True, exist_ok=True)
        (out_dir / parent / "train" / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / parent / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (out_dir / parent / "eval" / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / parent / "eval" / "labels").mkdir(parents=True, exist_ok=True)

    labeled_files = list(labeled_dir.glob("*.png"))

    if not ads_separation:
        # if no separation, all files are mixed and put in the "mixed" folder
        group_and_split_files(
            labeled_files,
            train_ratio,
            img_dir,
            out_dir / "mixed" / "train" / "images",
            out_dir / "mixed" / "train" / "labels",
            out_dir / "mixed" / "eval" / "images",
            out_dir / "mixed" / "eval" / "labels",
        )
    else:
        # separate ads-heavy and text-heavy files
        ads_heavy_files, text_heavy_files = separate_ads_text_heavy_files(
            labeled_files, ads_sep_conditions
        )

        print(f"Total ads-heavy files: {len(ads_heavy_files)}")
        print(f"Total text-heavy files: {len(text_heavy_files)}")

        group_and_split_files(
            ads_heavy_files,
            train_ratio,
            img_dir,
            out_dir / "ads" / "train" / "images",
            out_dir / "ads" / "train" / "labels",
            out_dir / "ads" / "eval" / "images",
            out_dir / "ads" / "eval" / "labels",
        )

        group_and_split_files(
            text_heavy_files,
            train_ratio,
            img_dir,
            out_dir / "text" / "train" / "images",
            out_dir / "text" / "train" / "labels",
            out_dir / "text" / "eval" / "images",
            out_dir / "text" / "eval" / "labels",
        )

    print(f"Data preparation completed.")


@click.option(
    "--img-dir",
    type=click.Path(
        writable=False,
        file_okay=False,
        dir_okay=True,
        exists=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Directory containing the downloaded Jingbao original images.",
)
@click.option(
    "--labeled-dir",
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
    "--train-ratio",
    type=float,
    default=0.8,
    help="Ratio of training data to total data.",
)
@click.option(
    "--ads-separation/--no-ads-separation",
    type=bool,
    default=False,
    help="Whether to separate ads from non-ads. If enabled, the data will be separated into two folders: one for training and evaluation of text-heavy, and one for training and evaluation of ads-heavy. If disabled, all data will be mixed together for training and evaluation.",
)
@click.option(
    "--out-dir",
    type=click.Path(
        writable=True,
        file_okay=False,
        dir_okay=True,
        exists=False,
        resolve_path=True,
        path_type=Path,
    ),
    help="Directory to save the data for training and evaluation. In case of ads separation, the output directory will contain two subdirectories: 'ads' and 'text', each containing 'train' and 'eval' subdirectories.",
)
@click.command
def prepare_data_cli(img_dir, labeled_dir, train_ratio, ads_separation, out_dir):
    prepare_data(
        img_dir,
        labeled_dir,
        train_ratio,
        ads_separation,
        out_dir,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Prepare data for training Eynollah."
    )
    argparser.add_argument(
        "--img-dir",
        type=Path,
        required=True,
        help="Directory containing the downloaded Jingbao original images.",
    )
    argparser.add_argument(
        "--labeled-dir",
        type=Path,
        required=True,
        help="Directory containing the labeled data.",
    )
    argparser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of training data to total data.",
    )
    argparser.add_argument(
        "--ads-separation",
        is_flag=True,
        help="Whether to separate ads from non-ads.",
    )
    argparser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to save the data for training and evaluation.",
    )
    args = argparser.parse_args()
    prepare_data(
        args.img_dir,
        args.labeled_dir,
        args.train_ratio,
        args.ads_separation,
        args.out_dir,
    )
