# download unseen data for testing

from pathlib import Path
import requests
import argparse

unseen_data = [
    "1919/04/jb_0015_1919-04-15_0001to0004.tif",
    "1919/04/jb_0015_1919-04-15_0002to0003.tif",
    "1920/01/jb_0103_1920-01-09_0001to0004.tif",
    "1920/01/jb_0103_1920-01-09_0002to0003.tif",
    "1920/03/jb_0118_1920-03-01_0001to0004.tif",
    "1920/03/jb_0118_1920-03-01_0002to0003.tif",
    "1920/06/jb_0150_1920-06-06_0001to0004.tif",
    "1920/06/jb_0150_1920-06-06_0002to0003.tif",
    "1920/08/jb_0173_1920-08-15_0001to0004.tif",
    "1920/08/jb_0174_1920-08-18_0002to0003.tif",
    "1920/12/jb_0215_1920-12-21_0001to0004.tif",
    "1920/12/jb_0215_1920-12-21_0002to0003.tif",
    "1921/02/jb_0234_1921-02-21_0001to0004.tif",
    "1921/02/jb_0234_1921-02-21_0002to0003.tif",
    "1922/02/jb_0351_1922-02-09_0001to0004.tif"
    "1922/02/jb_0351_1922-02-09_0002to0003.tif",
    "1923/02/jb_0472_1923-02-09_0001to0004.tif",
    "1923/02/jb_0472_1923-02-09_0002to0003.tif",
    "1924/01/jb_0584_1924-01-06_0001to0004.tif",
    "1924/01/jb_0588_1924-01-18_0002to0003.tif",
    "1925/01/jb_0703_1925-01-04_0001to0004.tif",
    "1925/01/jb_0703_1925-01-04_0002to0003.tif",
    "1926/01/jb_0824_1926-01-06_0001to0004.tif",
    "1926/01/jb_0827_1926-01-15_0002to0003.tif",
    # still need to check 1927 to 1940
]

PLACEHOLDER = "[placeholder]"

base_link = f"https://ecpo.cats.uni-heidelberg.de/fcgi-bin/iipsrv.fcgi?IIIF=/ecpo/images/jingbao/{PLACEHOLDER}/full/full/0/default.jpg"


def download_image(unseen_data: list, output_dir: Path, overwrite: bool = True):

    total = len(unseen_data)
    print(f"Total images to download: {total}")

    downloaded = 0
    failed = 0
    existed = 0
    overwrited = 0
    for item in unseen_data:
        img_link = base_link.replace(PLACEHOLDER, item)
        img_name = item.split("/")[-1].replace(".tif", ".png")

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


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Download unseen images from ecpo.cats."
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
    download_image(unseen_data, args.output_dir, args.overwrite)
