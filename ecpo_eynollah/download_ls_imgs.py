# download images from LS
from pathlib import Path
import json
import requests
import argparse
import click


def download_images(json_min_file: Path, output_dir: Path, overwrite: bool = True):

    with open(json_min_file, "r") as f:
        data = json.load(f)

    total = len(data)
    print(f"Total images to download: {total}")

    downloaded = 0
    failed = 0
    existed = 0
    overwrited = 0
    for item in data:
        img_link = item["image"]
        img_name = f"{item['name']}.png"

        # download image
        response = requests.get(img_link)
        if response.status_code == 200:
            if (output_dir / img_name).exists():
                existed += 1
                if not overwrite:
                    overwrited += 1
                    print(f"Image {img_name} already exists. Adding suffix '_dup'.")
                    img_name = f"{item['name']}_dup.png"

            with open(output_dir / img_name, "wb") as img_file:
                img_file.write(response.content)
                downloaded += 1
            print(f"Downloaded {img_name}")
        else:
            failed += 1
            print(f"Failed to download {img_name} from {img_link}")

    print(f"Download completed: {downloaded} succeeded, {failed} failed.")
    print(
        f"{existed} images already existed and {overwrited} were added with suffix '_dup'."
    )


@click.option(
    "--json-file",
    type=click.Path(
        writable=False,
        file_okay=True,
        dir_okay=False,
        exists=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Path to the JSON min file.",
)
@click.option(
    "--output-dir",
    type=click.Path(
        writable=True,
        file_okay=False,
        dir_okay=True,
        exists=False,
        resolve_path=True,
        path_type=Path,
    ),
    help="Directory to save downloaded images.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=True,
    help="Overwrite existing images if they exist.",
)
@click.command
def download_ls_imgs_cli(
    json_file: Path,
    output_dir: Path,
    overwrite: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    download_images(json_file, output_dir, overwrite)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Download images from LS JSON file.")
    parser.add_argument(
        "--json-file", type=Path, required=True, help="Path to the JSON min file."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save downloaded images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing images if they exist.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    download_images(args.json_file, args.output_dir, args.overwrite)
