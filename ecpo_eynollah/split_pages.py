"""
split_pages.py

Usage:
    python split_pages.py [--config config_file.json --tag unique_tag]

Notes:
- For each input image it will:
    1) rotate image but kep size unchanged
    2) compute a smoothed vertical projection and detect local minima (gutters)
    3) split into vertical segments at those minima; always covers full width
    4) save segments as <img_name>_pX.jpg

Default config in root/config/default_config.json
"""

# the below code use opencv python
# another option to consider is scantailor-advanced
# https://github.com/4lex4/scantailor-advanced?tab=readme-ov-file


import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from ecpo_eynollah import config_handler
from ecpo_eynollah import utils
from pathlib import Path
from typing import List, Tuple
import shutil

# ----------------------------
# CONFIG / TUNABLE PARAMETERS
# ----------------------------

# Projection smoothing: kernel width (in pixels) for 1D smoothing of column sums.
SMOOTH_KERNEL = 101  # must be odd; increase if noisy

# Local minima detection:
MIN_DISTANCE_BETWEEN_SPLITS = 1200  # pixels; minimum allowed distance between splits
MIN_DEPTH_FACTOR = (
    0.55  # minima must be this fraction below median projection to qualify
)

# If no minima detected, fallback to a single center split (two pages)
FALLBACK_TO_CENTER = True

# Output JPEG quality
JPEG_QUALITY = 95  # it can be a quality from 0 to 100 (the higher is the better)


# ----------------------------
# Step 1: rotate document and keep size
# ----------------------------
def rotate_img(img):
    # TODO
    pass


# ----------------------------
# Step 2: Find split columns (possibly >2)
# ----------------------------
def smooth_1d(signal: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    """Smooths a 1D signal using a simple moving average.

    Args:
        signal (np.ndarray): 1D array to smooth.
        kernel_size (int): Size of the smoothing kernel (must be odd).

    Returns:
        np.ndarray: Smoothed 1D array.
    """
    if kernel_size <= 1:
        return signal
    # ensure odd kernel size
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    # simple moving average via convolution
    kernel = np.ones(k, dtype=np.float32) / k
    padded = np.pad(signal, (k // 2, k // 2), mode="edge")
    # slide kernel over signal
    sm = np.convolve(padded, kernel, mode="valid")
    return sm


def convert_to_binary_morp(
    img: np.ndarray,
    fname: str,
    output_dir: Path,
    unique_tag: str,
    img_quality: int = 95,
    save_for_debug: bool = False,
) -> np.ndarray:
    """Convert image to binary morphed image for projection analysis.

    Args:
        img (np.ndarray): Input image in BGR format (H, W, 3).
        fname (str): filename of the image, used for saving debug images.
        output_dir (Path): directory to save debug images.
        unique_tag (str): unique tag to append to debug image filenames.
        img_quality (int): JPEG quality for saving debug images.
        save_for_debug (bool): whether to save intermediate images for debugging.

    Returns:
        np.ndarray: morphed binary image.
    """
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if save_for_debug:
        utils.save_jpeg(
            output_dir / f"debug_{fname}_gray_{unique_tag}.jpg",
            gray,
            quality=img_quality,
        )

    # blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # (5,5) is kernel for GaussianBlur
    if save_for_debug:
        utils.save_jpeg(
            output_dir / f"debug_{fname}_blur_{unique_tag}.jpg",
            blur,
            quality=img_quality,
        )

    # convert grayscale to black & white
    # adaptive threshold is calculated for each pixel based on local neighborhood
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 705, 15
    )  # maxValue=255, blockSize=35, C=15
    if save_for_debug:
        utils.save_jpeg(
            output_dir / f"debug_{fname}_th_{unique_tag}.jpg", th, quality=img_quality
        )

    # create kernel for morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    # fill small holes INSIDE the foreground objects
    # close = dilate -> erode
    # more info about morphological transformations:
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    morp = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    return morp


def find_split_columns(
    img_rect: np.ndarray, fname: str, unique_tag: str, **kwargs
) -> Tuple[List[int], bool]:
    """Returns a sorted list of x coordinates where splits should occur (columns).
    Return only the split columns (internal cuts).

    Args:
        img_rect (np.ndarray): Rectified image in BGR format (H, W, 3).
        fname (str): filename of the image, used for saving debug images.
        unique_tag (str): unique tag to append to debug image filenames.
        **kwargs: parameters from config, e.g.:
            smooth_kernel (int): kernel size for smoothing projection.
            min_distance_between_splits (int): minimum distance between splits.
            min_depth_factor (float): minimum depth factor for minima.
            fallback_to_center (bool): whether to fallback to center split if no minima found.
            jpeg_quality (int): JPEG quality for saving debug images.
            save_intermediate_images (bool): whether to save intermediate images for debugging.

    Returns:
        Tuple[List[int], bool]: (list of internal split columns, fallback used)
    """
    # config for debug saving
    output_dir = Path(kwargs.get("output_dir", "."))
    save_for_debug = kwargs.get("save_intermediate_images", False)
    img_quality = kwargs.get("jpeg_quality", 95)

    # crop image, skip to keep the original size
    _, w = img_rect.shape[:2]
    left = 0

    if kwargs.get("convert_bw", True):
        # convert to black and white morphed image
        morp = convert_to_binary_morp(
            img_rect,
            fname,
            output_dir,
            unique_tag,
            img_quality=img_quality,
            save_for_debug=save_for_debug,
        )
    else:
        # use grayscale directly
        morp = cv2.cvtColor(img_rect, cv2.COLOR_BGR2GRAY)
    if save_for_debug:
        utils.save_jpeg(
            output_dir / f"debug_{fname}_morp_{unique_tag}.jpg",
            morp,
            quality=img_quality,
        )

    # projection: sum of intensities per column
    proj = np.sum(morp.astype(np.float32), axis=0)
    # We expect gutter to be darker -> local minima in proj
    # Smooth the projection
    sm = smooth_1d(proj, kernel_size=kwargs.get("smooth_kernel", 51))
    # Normalize
    sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-9)

    # Find local minima: compare each point to neighbors within +/-1
    minima_idx = []
    for i in range(1, len(sm) - 1):
        if sm[i] < sm[i - 1] and sm[i] < sm[i + 1]:
            minima_idx.append(i)

    # Filter minima by depth threshold relative to median
    median_val = np.median(sm)
    candidates = []
    for i in minima_idx:
        val = sm[i]
        if val < median_val * kwargs.get("min_depth_factor", 0.55):
            candidates.append(i)

    # Convert candidate indices to absolute columns in rectified image
    candidates = [int(left + i) for i in candidates]

    # Enforce minimum distance between splits
    candidates_sorted = sorted(candidates)
    filtered = []
    last = -10000
    for c in candidates_sorted:
        if c - last >= kwargs.get("min_distance_between_splits", 600):
            filtered.append(c)
            last = c
        else:
            # choose deeper minima among close ones (optional heuristic)
            # skip for simplicity
            pass

    # If no splits were found, fallback to center split (one internal split)
    fallback = False
    if len(filtered) == 0 and kwargs.get("fallback_to_center", True):
        mid = w // 2
        filtered = [mid]
        fallback = True

    # Return internal split columns only (not 0 or w)
    return filtered, fallback


# ----------------------------
# Step 3: Slice & save
# ----------------------------
def slice_and_save(
    img_rect: np.ndarray, splits_internal: List[int], output_dir: Path, fname: str
):
    """Slice the rectified image at the given split columns and save segments.
    Segments are saved as <fname>_pX.jpg where X is the segment index starting from 0.

    Args:
        img_rect (np.ndarray): Rectified image in BGR format (H, W, 3).
        splits_internal (List[int]): List of internal split columns (x coordinates).
        output_dir (Path): Directory to save the segments.
        fname (str): filename of the image, used for naming output segments.
    """
    _, w = img_rect.shape[:2]
    cuts = [0] + splits_internal + [w]
    saved = []
    for i in range(len(cuts) - 1):
        x0, x1 = cuts[i], cuts[i + 1]
        # ensure non-empty
        if x1 - x0 <= 10:
            continue  # too narrow, skip

        # create an image of same size but mask other areas except the segment
        seg = img_rect[:, x0:x1]
        out_img = np.full_like(img_rect, 255)
        out_img[:, x0:x1] = seg

        # save the imgage with only the segment visible
        outname = f"{fname}_p{i}.jpg"
        outpath = output_dir / outname
        utils.save_jpeg(outpath, out_img, quality=JPEG_QUALITY)
        saved.append((outpath, x0, x1))

    return saved


# ----------------------------
# Main batch function
# ----------------------------
def process_folder(config_path: str | None = None, unique_tag: str | None = None):
    """Process a folder of images to split pages according to the config.

    Args:
        config_path (str | None): Path to the config file.
            If None, use default config.
        unique_tag (str | None): Unique tag to append to output files.
    """
    # load config
    config, _ = config_handler.load_config(config_path=config_path, new_config=None)
    input_dir = Path(config.get("input_dir"))
    gutter_detect_config = config.get("gutter_detection")

    if not input_dir or input_dir.is_file():
        raise ValueError("Input directory is not specified or is a file.")

    if not gutter_detect_config:
        raise ValueError("Gutter detection config is missing.")

    output_dir = Path(gutter_detect_config.get("output_dir"))
    jpeg_quality = gutter_detect_config.get("jpeg_quality", 95)
    save_intermediate_images = gutter_detect_config.get(
        "save_intermediate_images", True
    )

    # ensure the output dir exists
    utils.ensure_dir(output_dir)

    # prepare tag and save used config
    if unique_tag is None:
        unique_tag = utils.generate_unique_tag()
    config_handler.save_config_to_file(
        config, output_dir, file_name=f"used_config_{unique_tag}.json"
    )
    # save a copy of this script
    shutil.copy2(Path(__file__), output_dir / f"split_pages_{unique_tag}.py")

    # collect all files in input dir
    files = sorted(
        f
        for f in input_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    )

    log_lines = []
    fallback_count = 0
    fallback_files = []
    for _, fpath in enumerate(tqdm(files, desc="files")):
        try:
            img = utils.load_image(fpath)
        except Exception as e:
            print(f"ERROR loading {fpath}: {e}")
            continue

        # step 1: rotate
        # rect = rotate_img(img)
        rect = img  # disable this step for testing
        if save_intermediate_images:
            utils.save_jpeg(
                output_dir / f"debug_{fpath.stem}_rect_{unique_tag}.jpg",
                rect,
                quality=jpeg_quality,
            )

        # step 2: find splits
        splits, fallback = find_split_columns(
            rect, fpath.stem, unique_tag, **gutter_detect_config
        )
        if fallback:
            fallback_count += 1
            fallback_files.append(fpath.name)

        # step 3: slice & save
        saved = slice_and_save(rect, splits, output_dir, fpath.stem)

        # log
        split_str = ",".join(str(x) for x in splits)
        log_lines.append(
            f"{fpath.name}\t{len(saved)}\t{split_str}\t{'x' if fallback else ''}"
        )

    # write log
    with open(
        os.path.join(output_dir, f"split_log_{unique_tag}.csv"), "w", encoding="utf8"
    ) as f:
        f.write("input_file\tsegments\tsplits_internal\tfallback\n")
        for L in log_lines:
            f.write(L + "\n")

    # display summary
    print(f"Processed {len(files)} files.")
    print(f"Output saved to: {output_dir}")
    print(f"Fallback to center split used in {fallback_count} files.")
    if fallback_count > 0:
        print("Files with fallback:")
        for fn in fallback_files:
            print(f" - {fn}")


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Batch-split images into page segments.")
    p.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        help="Path to the config file.",
    )

    p.add_argument(
        "-t",
        "--tag",
        type=str,
        required=False,
        help="Unique tag to append to output files.",
    )

    args = p.parse_args()
    process_folder(args.config, args.tag)
