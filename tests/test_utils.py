import pytest
from datetime import datetime
from ecpo_eynollah import utils
import numpy as np
import cv2


def test_ensure_dir(tmp_path):
    dir_path = tmp_path / "new_dir"
    assert not dir_path.exists()
    utils.ensure_dir(dir_path)
    assert dir_path.exists()
    assert dir_path.is_dir()


def test_load_image(tmp_path):
    # Create a simple image and save it
    img_path = tmp_path / "test_image.jpg"
    img = 255 * np.ones((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    # Load the image using the utility function
    loaded_img = utils.load_image(img_path)
    assert loaded_img.shape == (100, 100, 3)
    assert np.array_equal(loaded_img, img)

    # invalid case
    with pytest.raises(ValueError):
        utils.load_image(tmp_path / "non_existent.jpg")

    tmp_file = tmp_path / "not_an_image.txt"
    tmp_file.write_text("This is not an image.")
    with pytest.raises(ValueError):
        utils.load_image(tmp_file)


def test_save_jpeg_invalid_path():
    img = 255 * np.ones((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        utils.save_jpeg("", img)

    with pytest.raises(ValueError):
        utils.save_jpeg("image.jpg", img, quality=150)


def test_save_jpeg(tmp_path):
    img = 255 * np.ones((100, 100, 3), dtype=np.uint8)
    save_path = tmp_path / "saved_image.jpg"

    utils.save_jpeg(save_path, img, quality=90)
    assert save_path.exists()

    loaded_img = cv2.imread(str(save_path))
    assert loaded_img.shape == (100, 100, 3)
    assert np.array_equal(loaded_img, img)


def test_get_img_size():
    # valid image
    img = 255 * np.ones((200, 150, 3), dtype=np.uint8)
    h, w = utils.get_img_size(img)
    assert h == 200
    assert w == 150

    # invalid image
    with pytest.raises(ValueError):
        utils.get_img_size(None)
    with pytest.raises(ValueError):
        utils.get_img_size(np.array([1]))


def test_generate_unique_tag():
    unique_tag = utils.generate_unique_tag()
    assert isinstance(unique_tag, str)
    assert (
        len(unique_tag.split("_")) == 2
    )  # should be in the format "YYYYMMDD-HHMMSS_hostname"

    # Check if the timestamp is in the correct format
    datetime_part, hostname_part = unique_tag.split("_")
    assert "ts" in datetime_part  # should start with "ts"
    datetime.strptime(datetime_part[2:], "%Y%m%d-%H%M%S")

    # Check if the hostname is a valid string
    assert "h" in hostname_part  # should start with "h"
    assert isinstance(hostname_part, str) and len(hostname_part) > 0
