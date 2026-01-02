# download images from LS
from pathlib import Path
import json
import requests
import argparse


def download_image(json_min_file: Path, output_dir: Path):

    with open(json_min_file, "r") as f:
        data = json.load(f)

    total = len(data)
    print(f"Total images to download: {total}")

    downloaded = 0
    failed = 0
    exist = 0
    for item in data:
        img_link = item["image"]
        img_name = f"{item['name']}.png"

        # download image
        response = requests.get(img_link)
        if response.status_code == 200:
            if (output_dir / img_name).exists():
                exist += 1
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
    print(f"{exist} images already existed and were added with suffix '_dup'.")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Download images from LS JSON file.")
    parser.add_argument(
        "--json_file", type=Path, required=True, help="Path to the JSON min file."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save downloaded images.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    download_image(args.json_file, args.output_dir)
