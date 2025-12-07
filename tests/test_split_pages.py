import pytest
from ecpo_eynollah import split_pages as sp
from PIL import Image, ImageDraw, ImageFont
import numpy as np


@pytest.fixture
def get_img():
    # Image size and gap configuration
    img_w, img_h = 1000, 800
    gap = 300

    # Create blank white image
    img = Image.new("RGB", (img_w, img_h), color="white")
    draw = ImageDraw.Draw(img)

    # Text content
    text_left = (
        "This is the LEFT text block.\n"
        "It is used to test OCR detection.\n"
        "There will be a blank space in the center."
    )

    text_right = (
        "This is the RIGHT text block.\n"
        "It mirrors the layout of the left side.\n"
        "Both sides leave a 300 px gap."
    )

    # Load font (optional)
    font = ImageFont.load_default()

    # Measure each text block
    bbox_left = draw.multiline_textbbox((0, 0), text_left, font=font, align="left")
    bbox_right = draw.multiline_textbbox((0, 0), text_right, font=font, align="left")

    text_w_left = bbox_left[2] - bbox_left[0]
    text_h_left = bbox_left[3] - bbox_left[1]
    text_w_right = bbox_right[2] - bbox_right[0]
    text_h_right = bbox_right[3] - bbox_right[1]

    # Compute positions
    total_text_width = text_w_left + gap + text_w_right
    start_x = (img_w - total_text_width) / 2  # horizontally center the pair
    y = (img_h - max(text_h_left, text_h_right)) / 2  # vertical center

    # Draw left text
    draw.multiline_text((start_x, y), text_left, fill="black", font=font, align="left")

    # Draw right text
    draw.multiline_text(
        (start_x + text_w_left + gap, y),
        text_right,
        fill="black",
        font=font,
        align="left",
    )

    return img


@pytest.fixture
def get_img_path(get_img, tmp_path):
    img_path = tmp_path / "test_image.png"
    get_img.save(img_path, format="PNG")
    return img_path


def test_get_text_detections_paddleocr(get_img_path):
    ocr_model = "PP-OCRv5_server_det"
    in_img, det_polys = sp.get_text_detections_paddleocr(
        str(get_img_path), ocr_model, device="cpu"
    )
    assert in_img.shape == (800, 1000, 3)
    assert len(det_polys) > 0


def test_compute_signal(get_img_path):
    in_img, det_polys = sp.get_text_detections_paddleocr(
        str(get_img_path), ocr_model="PP-OCRv5_server_det", device="cpu"
    )
    signal, mask_array = sp.compute_signal(in_img, det_polys, proj_func=np.mean)
    assert len(signal) == in_img.shape[1]  # signal length should match image width
    assert all(0 <= val <= 255 for val in signal)  # signal values should be in [0, 255]
    assert mask_array.shape == (in_img.shape[0], in_img.shape[1])


def test_find_split_points(get_img_path):
    in_img, det_polys = sp.get_text_detections_paddleocr(
        str(get_img_path), ocr_model="PP-OCRv5_server_det", device="cpu"
    )
    signal, _ = sp.compute_signal(in_img, det_polys, proj_func=np.mean)
    split_points, fallback, org_bkps = sp.find_split_points(
        signal,
        num_bkps=4,
        close_thres=0.0,
        num_segments=3,
        fallback=True,
    )
    assert len(split_points) == 2
    assert fallback is False
    assert len(org_bkps) == 4


def test_slice_and_save(get_img_path, tmp_path):
    in_img, det_polys = sp.get_text_detections_paddleocr(
        str(get_img_path), ocr_model="PP-OCRv5_server_det", device="cpu"
    )
    signal, _ = sp.compute_signal(in_img, det_polys, proj_func=np.mean)
    split_points, fallback, org_bkps = sp.find_split_points(
        signal,
        num_bkps=4,
        close_thres=0.0,
        num_segments=3,
        fallback=True,
    )

    # save as 3 segments
    out_dir = tmp_path / "output_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    sp.slice_and_save(
        in_img,
        split_points,
        out_dir,
        fname="test_image",
        unique_tag="test",
        segment_size=100,
        jpeg_quality=95,
    )

    # Check if images are saved
    saved_files = list(out_dir.glob("test_image_*.jpg"))
    assert len(saved_files) == 3  # Should save 3 segments

    # save as 2 segments
    out_dir_2seg = tmp_path / "output_images_2seg"
    out_dir_2seg.mkdir(parents=True, exist_ok=True)

    sp.slice_and_save(
        in_img,
        split_points,
        out_dir_2seg,
        fname="test_image",
        unique_tag="test",
        segment_size=300,
        jpeg_quality=95,
    )

    # Check if images are saved
    saved_files_2seg = list(out_dir_2seg.glob("test_image_*.jpg"))
    assert len(saved_files_2seg) == 2  # Should save 2 segments
