# download unseen data for testing

from pathlib import Path
import requests
import argparse
import click

non_jingbao = [
    "ban_yue/volume001/issue0001/by_vol001_issue0001_0009.tif",
    "funu_wenhua_zstk/volume001/issue0002/fnwhzstk_vol001_issue0002_0001to0002.tif",
    "li_bao_1/1935/09/lb1_0005_1935-09-24_0001to0004.tif",
    "san_si_ju_wu_ri_kan/1926/09/ssjwrk_0002_1926-09-12_0001to0004.tif",
    "shehui_ribao/1941/03/shrb_3736_1941-03-15_0001to0004.tif",
    "shehui_ribao/1941/03/shrb_3736_1941-03-15_0002to0003.tif",
    "shijie_fanhuabao/1901/09/sjfhb_0174_1901-09-27_0001to0002.tif",
    "tuhua_jubao/1913/03/thjb_0115_1913-03-13_0001.tif",
    "wei_sheng_bao/1927/12/wsb_0002_1927-12-17_0001.tif",
    "xi_bao/1946/10/xb_0005_1946-10-23_0001to0004.tif",
]

PLACEHOLDER = "[placeholder]"

base_link = f"https://ecpo.cats.uni-heidelberg.de/fcgi-bin/iipsrv.fcgi?IIIF=/ecpo/images/{PLACEHOLDER}/full/full/0/default.jpg"


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
                    img_name = f"{img_name.split('.')[0]}_dup.png"

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
def download_non_jingbao_cli(output_dir: Path, overwrite: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    download_images(non_jingbao, output_dir, overwrite)


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
    download_images(non_jingbao, args.output_dir, args.overwrite)
