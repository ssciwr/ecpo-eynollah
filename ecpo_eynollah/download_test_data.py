# download unseen data for testing

from pathlib import Path
import requests
import argparse
import click

unseen_data = [
    "1919/04/jb_0015_1919-04-15_0001to0004.tif",
    "1919/04/jb_0015_1919-04-15_0002to0003.tif",
    "1920/01/jb_0103_1920-01-09_0001to0004.tif",
    "1920/01/jb_0103_1920-01-09_0002to0003.tif",
    "1921/02/jb_0234_1921-02-21_0001to0004.tif",
    "1921/02/jb_0234_1921-02-21_0002to0003.tif",
    "1922/02/jb_0351_1922-02-09_0001to0004.tif",
    "1922/02/jb_0351_1922-02-09_0002to0003.tif",
    "1923/02/jb_0472_1923-02-09_0001to0004.tif",
    "1923/02/jb_0472_1923-02-09_0002to0003.tif",
    "1924/01/jb_0584_1924-01-06_0001to0004.tif",
    "1924/01/jb_0588_1924-01-18_0002to0003.tif",
    "1925/01/jb_0703_1925-01-04_0001to0004.tif",
    "1925/01/jb_0703_1925-01-04_0002to0003.tif",
    "1926/01/jb_0824_1926-01-06_0001to0004.tif",
    "1926/01/jb_0827_1926-01-15_0002to0003.tif",
    "1927/01/jb_0942_1927-01-03_0001to0004.tif",
    "1927/01/jb_0944_1927-01-09_0002to0003.tif",
    "1928/01/jb_1062_1928-01-09_0001to0004.tif",
    "1928/01/jb_1068_1928-01-30_0002to0003.tif",
    "1929/01/jb_1182_1929-01-12_0001to0004.tif",
    "1929/01/jb_1182_1929-01-12_0002to0003.tif",
    "1930/01/jb_1296_1930-01-01_0001to0004.tif",
    "1930/02/jb_1305_1930-02-06_0002to0003.tif",
    "1931/01/jb_1414_1931-01-09_0001to0004.tif",
    "1931/01/jb_1414_1931-01-09_0002to0003.tif",
    "1932/01/jb_1533_1932-01-15_0001to0004.tif",
    "1932/01/jb_1531_1932-01-09_0002to0003.tif",
    "1933/01/jb_1714_1933-01-19_0001to0004.tif",
    "1933/01/jb_1714_1933-01-19_0002to0003.tif",
    "1934/02/jb_2079_1934-02-02_0001to0004.tif",
    "1934/02/jb_2079_1934-02-02_0002to0003.tif",
    "1935/01/jb_2408_1935-01-12_0001to0004.tif",
    "1935/01/jb_2408_1935-01-12_0002to0003.tif",
    "1936/05/jb_2865_1936-05-06_0001to0004.tif",
    "1936/05/jb_2869_1936-05-10_0002to0003.tif",
    "1937/01/jb_3104_1937-01-05_0001to0004.tif",
    "1937/01/jb_3106_1937-01-07_0002to0003.tif",
    "1938/02/jb_3428_1938-02-04_0001to0004.tif",
    "1938/02/jb_3429_1938-02-05_0002to0003.tif",
    "1939/01/jb_3759_1939-01-02_0001to0004.tif",
    "1939/01/jb_3759_1939-01-02_0002to0003.tif",
    "1940/01/jb_4023_1940-01-02_0001to0004.tif",
    "1940/01/jb_4025_1940-01-04_0002to0003.tif",
    "1940/01/jb_4025_1940-01-04_0005.tif",
    "1940/01/jb_4027_1940-01-06_0006.tif",
]

PLACEHOLDER = "[placeholder]"

base_link = f"https://ecpo.cats.uni-heidelberg.de/fcgi-bin/iipsrv.fcgi?IIIF=/ecpo/images/jingbao/{PLACEHOLDER}/full/full/0/default.jpg"


def download_images(unseen_data: list, output_dir: Path, overwrite: bool = True):

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
def download_test_data_cli(output_dir: Path, overwrite: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    download_images(unseen_data, output_dir, overwrite)


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
    download_images(unseen_data, args.output_dir, args.overwrite)
