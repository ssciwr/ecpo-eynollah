"""
split_pages.py

Usage:
    python split_pages.py [--config config_file.json --tag unique_tag]

Notes:
- For each input image it will:
    1) use PaddleOCR to find text dection on the image
    2) compute signal (column-wise projection) to find vertical split points
        2.1) based on the text detection result to mask text areas, masked areas are white, background is black
        2.2) compute column-wise projection on the masked image, i.e. mean of pixel values per column (signal)
    3) find vertical split points based on the signal
        3.1) use Dynamic Programming to find vertical breakpoints of significant gaps
        3.2) find refined points between those breakpoints that their signal near zero (black), i.e. no text there
        3.3) only consider points near the center to ensure we have num_segments - 1 splits
    4) split into vertical segments at those refined points; always covers full width
    5) save segments as <img_name>_pX.jpg

Default config in root/config/default_config.json
"""

# the below code use opencv python
# another option to consider is scantailor-advanced
# https://github.com/4lex4/scantailor-advanced?tab=readme-ov-file


import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import argparse
from ecpo_eynollah import config_handler
from ecpo_eynollah import utils
from pathlib import Path
from typing import List, Tuple, Callable, Dict
import shutil
from paddleocr import TextDetection
from PIL import Image, ImageDraw
import ruptures as rpt


# ----------------------------
# Step 1: use PaddleOCR to detect text
# ----------------------------
def get_text_detections_paddleocr(
    img_path: Path, ocr_model: str, device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """Get text detections using PaddleOCR.

    Note: using PaddleOCR TextDetection predict directly with file path yields better
    Dynamic Programming results than using with a pre-loaded image (as numpy array).

    Args:
        img_path (Path): Path to the input image.
        ocr_model (str): PaddleOCR model name.
        device (str): Device to run the model on. e.g. "cpu", "gpu", "gpu:0".

    Returns:
        Tuple[np.ndarray, np.ndarray]: The input image and detected text polygons.
    """
    model = TextDetection(model_name=ocr_model, device=device)
    det_result = model.predict(img_path, batch_size=1)

    first_result = det_result[0]  # only one image
    in_img = first_result.get("input_img")
    det_polys = first_result.get("dt_polys")

    return in_img, det_polys


# ----------------------------
# Step 2: compute signal (column-wise projection) with text areas masked
# ----------------------------
def compute_signal(
    img: np.ndarray,
    dt_polys: np.ndarray,
    proj_func: Callable[[npt.ArrayLike], float] = np.mean,
) -> np.ndarray:
    """Create a mask from text detection polygons and compute column-wise projection signal.

    Args:
        img (np.ndarray): Input image in BGR format (H, W, 3).
        dt_polys (np.ndarray): Detected text polygons, shape (N, 4, 2).
        proj_func (Callable): Function to compute projection, e.g. np.sum or np.mean.

    Returns:
        np.ndarray: Column-wise projection signal.
    """
    # create a mask with background black (0)
    h, w = img.shape[:2]
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    # draw text areas as white (255)
    for poly in dt_polys:
        points = [tuple(point) for point in poly]
        draw.polygon(points, fill=255)

    # compute signal
    mask_array = np.array(mask)
    signal = proj_func(mask_array, axis=0)  # column-wise

    return signal


# ----------------------------
# Step 3: find split points using Dynamic Programming
# ----------------------------
def find_split_points(
    signal: np.ndarray,
    num_bkps: int = 4,
    close_thres: float = 1e-3,
    num_segments: int = 2,
    fallback: bool = True,
) -> Tuple[List[int], bool]:
    """Find breakpoints in the signal using Dynamic Programming,
    and return points within the resulting segments where the signal is close to zero.

    Args:
        signal (np.ndarray): 1D array representing the signal.
        num_bkps (int): Number of breakpoints to find with DP.
        close_thres (float): Threshold to consider a point as "close to zero".
        num_segments (int): Number of segments (pages) to split the image into.
        fallback (bool): Whether to fallback to center split if no breakpoints found.

    Returns:
        Tuple[List[int], bool]: List of refined breakpoints
            and a flag indicating if fallback was used.
    """
    # use ruptures to find breakpoints
    algo = rpt.Dynp(model="l2").fit(signal)
    bkps = algo.predict(n_bkps=num_bkps)
    bkps = bkps[:-1]  # remove the last point which is length of signal

    # find near-zero points within segments
    near_zero_points = []
    for start, stop in zip(bkps[:-1], bkps[1:]):
        segment = signal[start:stop]
        near_zero_mask = np.isclose(segment, 0.0) | (segment <= close_thres)
        # got index where mask is True
        near_zero_ps = np.where(near_zero_mask)[0] + start
        near_zero_points.extend(near_zero_ps.tolist())

    # group near-zero points into groups of continuous indices
    groups = []
    current_group = []
    for i in range(len(near_zero_points)):
        if i == 0:
            current_group.append(near_zero_points[i])
        else:
            if near_zero_points[i] == near_zero_points[i - 1] + 1:
                current_group.append(near_zero_points[i])
            else:
                groups.append(current_group)
                current_group = [near_zero_points[i]]
    if current_group:
        groups.append(current_group)

    # record only start and end of each group as refined breakpoints
    refined_bkps = []
    for group in groups:
        refined_bkps.append(group[0])
        refined_bkps.append(group[-1])

    # get only breakpoints near the center to make sure we have num_segments - 1 splits
    center = signal.shape[0] / 2
    filtered_bkps = sorted(
        refined_bkps,
        key=lambda x: abs(x - center),
    )[: num_segments - 1]
    filtered_bkps = sorted(filtered_bkps)

    use_fallback = False
    if not filtered_bkps and fallback:
        # fallback to center split
        w = signal.shape[0]
        filtered_bkps = [w // 2]
        use_fallback = True

    return filtered_bkps, use_fallback


# ----------------------------
# Step 4 & 5: Slice & save
# ----------------------------
def slice_and_save(
    img: np.ndarray,
    splits_internal: List[int],
    output_dir: Path,
    fname: str,
    unique_tag: str,
    jpeg_quality: int = 95,
):
    """Slice the rectified image at the given split columns and save segments.
    Segments are saved as <fname>_pX.jpg where X is the segment index starting from 0.

    Args:
        img (np.ndarray): Rectified image in BGR format (H, W, 3).
        splits_internal (List[int]): List of internal split columns (x coordinates).
        output_dir (Path): Directory to save the segments.
        fname (str): filename of the image, used for naming output segments.
        unique_tag (str): unique tag to append to output image filenames.
        jpeg_quality (int): JPEG quality for saving images.
    """
    w = img.shape[1]
    cuts = [0] + splits_internal + [w]
    saved = []
    for i in range(len(cuts) - 1):
        x0, x1 = cuts[i], cuts[i + 1]
        # ensure non-empty
        if x1 - x0 <= 10:
            continue  # too narrow, skip

        # create an image of same size but mask other areas except the segment
        seg = img[:, x0:x1]
        out_img = np.full_like(img, 255)
        out_img[:, x0:x1] = seg

        # save the imgage with only the segment visible
        outname = f"{fname}_p{i}_{unique_tag}.jpg"
        outpath = output_dir / outname
        utils.save_jpeg(outpath, out_img, quality=jpeg_quality)
        saved.append((outpath, x0, x1))

    return saved


# ----------------------------
# Main batch function
# ----------------------------
def process_folder(
    config_path: str | None = None,
    new_config: Dict | None = None,
    unique_tag: str | None = None,
):
    """Process a folder of images to split pages according to the config.

    Args:
        config_path (str | None): Path to the config file.
            If None or "default", use default config.
        new_config (Dict | None): New config to override the loaded config.
        unique_tag (str | None): Unique tag to append to output files.
            If None, tag will be "ts{YYYYMMDD-HHMMSS}_h{hostname}".
    """
    # load config
    config, _ = config_handler.load_config(
        config_path=config_path, new_config=new_config
    )
    input_dir = Path(config.get("input_dir"))
    gutter_detect_config = config.get("gutter_detection")

    if not input_dir or input_dir.is_file():
        raise ValueError("Input directory is not specified or is a file.")

    if not gutter_detect_config:
        raise ValueError("Gutter detection config is missing.")

    output_dir = Path(gutter_detect_config.get("output_dir"))
    ocr_model = gutter_detect_config.get("ocr_model", "PP-OCRv5_server_det")
    device = gutter_detect_config.get("device", "cpu")
    proj_func_name = gutter_detect_config.get("proj_func", "mean")
    number_breakpoints = gutter_detect_config.get("number_breakpoints", 4)
    close_threshold = gutter_detect_config.get("close_threshold", 1e-3)
    fallback_to_center = gutter_detect_config.get("fallback_to_center", True)
    num_segments = gutter_detect_config.get("num_segments", 2)
    jpeg_quality = gutter_detect_config.get("jpeg_quality", 95)

    proj_func_map = {
        "mean": np.mean,
        "sum": np.sum,
        "max": np.max,
        "min": np.min,
    }

    proj_func = proj_func_map.get(proj_func_name, np.mean)

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
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    log_lines = []
    fallback_count = 0
    fallback_files = []
    for _, fpath in enumerate(tqdm(files, desc="files")):

        # step 1: use PaddleOCR to detect text
        in_img, dt_polys = get_text_detections_paddleocr(
            str(fpath), ocr_model=ocr_model, device=device
        )

        # step 2: compute signal
        signal = compute_signal(in_img, dt_polys, proj_func=proj_func)

        # step 3: find split points
        points, fallback = find_split_points(
            signal,
            num_bkps=number_breakpoints,
            close_thres=close_threshold,
            num_segments=num_segments,
            fallback=fallback_to_center,
        )
        if fallback:
            fallback_count += 1
            fallback_files.append(fpath.name)

        # step 3: slice & save
        saved = slice_and_save(
            in_img,
            points,
            output_dir,
            fpath.stem,
            unique_tag=unique_tag,
            jpeg_quality=jpeg_quality,
        )

        # log
        split_str = ",".join(str(x) for x in points)
        log_lines.append(
            f"{fpath.name}\t{len(saved)}\t{split_str}\t{'x' if fallback else ''}"
        )

    # write log
    with open(output_dir / f"split_log_{unique_tag}.csv", "w", encoding="utf8") as f:
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
    process_folder(
        config_path=args.config, unique_tag=args.tag
    )  # omit new_config for CLI
