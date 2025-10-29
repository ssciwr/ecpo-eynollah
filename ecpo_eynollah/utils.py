from pathlib import Path
import numpy as np
import cv2
from datetime import datetime
import socket


def ensure_dir(path: Path):
    """Ensures that the directory at the given path exists.
    If it does not exist, it is created.

    Args:
        path (Path): Path to the directory.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    """Loads an image from the given path.

    Args:
        path (Path): Path to the image file.

    Returns:
        np.ndarray: Loaded image in BGR format (H, W, 3).

    Raises:
        ValueError: If the image file path is invalid,
            or if the image cannot be loaded.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except Exception as e:
        raise ValueError(f"File path {path} is invalid.") from e

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image from path: {path}")
    return img


def save_jpeg(path: Path, img: np.ndarray, quality: int = 95):
    """Saves an image as JPEG to the given path.

    Args:
        path (Path): Path to save the image file.
        img (np.ndarray): Image to save in BGR format (H, W, 3).
        quality (int): JPEG quality (0-100). Default is 95.
    """
    if not path:
        raise ValueError("Path to save image is empty.")

    if quality < 0 or quality > 100:
        raise ValueError("Quality must be between 0 and 100.")

    ext = Path(path).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        # if no valid extension, default to .jpg
        ext = ".jpg"
        path = path + ext

    params = (
        [int(cv2.IMWRITE_JPEG_QUALITY), quality] if ext in [".jpg", ".jpeg"] else []
    )
    _, enc = cv2.imencode(ext, img, params)

    enc.tofile(path)


def get_img_size(img: np.ndarray) -> tuple[int, int]:
    """Gets the width and height of the image.

    Args:
        img (np.ndarray): Image in BGR format (H, W, 3).
    """
    invalid_img = img is None or len(img.shape) < 2
    if invalid_img:
        raise ValueError("Invalid image provided.")

    h, w = img.shape[:2]
    return h, w


def generate_unique_tag() -> str:
    """Generate a unique tag based on the current timestamp and hostname.
    This will be used to identify different runs.

    Returns:
        str: A unique tag in the format "YYYYMMDD-HHMMSS_hostname".
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    hostname = socket.gethostname()
    return f"ts{timestamp}_h{hostname}"
