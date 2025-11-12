import pytest
import json
from pathlib import Path
from ecpo_eynollah import config_handler
from datetime import datetime
from tests.conftest import get_files


def test_is_non_empty_file(tmp_path):
    file_path = tmp_path / "test_file.txt"
    # file is not created yet
    assert config_handler.is_non_empty_file(file_path) is False

    # create an empty file
    file_path.touch()
    assert config_handler.is_non_empty_file(file_path) is False

    # create a non-empty file
    file_path.write_text("test")
    assert config_handler.is_non_empty_file(file_path) is True


def test_is_valid_config():
    config = {"input_dir": "data"}
    assert config_handler.is_valid_config(config, None) is True

    config = {"invalid_key": "data"}
    assert config_handler.is_valid_config(config, None) is False

    config = {
        "gutter_detection": {
            "output_dir": "./data/out",
            "ocr_model": "PP-OCRv5_server_det",
            "device": "cpu",
            "proj_func": "mean",
            "number_breakpoints": 4,
            "close_threshold": 0.001,
            "fallback_to_center": True,
            "num_segments": 2,
            "segment_size": 300,
            "jpeg_quality": 95,
        }
    }
    assert config_handler.is_valid_config(config, None) is True
    assert (
        config_handler.is_valid_config(config["gutter_detection"], "gutter_detection")
        is True
    )

    config = {"gutter_detection": {"output_dir": True}}
    assert config_handler.is_valid_config(config, None) is False

    config = {"ocr_model": True}
    assert config_handler.is_valid_config(config, "gutter_detection") is False

    config = {"device": 1}
    assert config_handler.is_valid_config(config, "gutter_detection") is False

    config = {"proj_func": "invalid"}
    assert config_handler.is_valid_config(config, "gutter_detection") is False
    config = {"proj_func": 1}
    assert config_handler.is_valid_config(config, "gutter_detection") is False

    config = {"number_breakpoints": "4"}
    assert config_handler.is_valid_config(config, "gutter_detection") is False
    config = {"number_breakpoints": 0}
    assert config_handler.is_valid_config(config, "gutter_detection") is False

    config = {"close_threshold": "0.001"}
    assert config_handler.is_valid_config(config, "gutter_detection") is False
    config = {"close_threshold": -0.1}
    assert config_handler.is_valid_config(config, "gutter_detection") is False

    config = {"fallback_to_center": "True"}
    assert config_handler.is_valid_config(config, "gutter_detection") is False

    config = {"num_segments": "2"}
    assert config_handler.is_valid_config(config, "gutter_detection") is False
    config = {"num_segments": 0}
    assert config_handler.is_valid_config(config, "gutter_detection") is False

    config = {"segment_size": "300"}
    assert config_handler.is_valid_config(config, "gutter_detection") is False
    config = {"segment_size": 0}
    assert config_handler.is_valid_config(config, "gutter_detection") is False

    config = {"jpeg_quality": "95"}
    assert config_handler.is_valid_config(config, "gutter_detection") is False
    config = {"jpeg_quality": 150}
    assert config_handler.is_valid_config(config, "gutter_detection") is False


def test_update_new_config_empty():
    updated = config_handler._update_new_config({"test": "test"}, {}, parent_key=None)
    assert updated is False

    with pytest.raises(ValueError):
        config_handler._update_new_config({}, {"test": "test"}, parent_key=None)


def test_update_new_config_not_updated():
    # invalid key
    with pytest.warns(UserWarning):
        updated = config_handler._update_new_config(
            {"input_dir": "data"}, {"test": "test"}, parent_key=None
        )
    assert updated is False

    # invalid structure
    with pytest.warns(UserWarning):
        updated = config_handler._update_new_config(
            {"input_dir": "data"}, {"input_dir": 1}, parent_key=None
        )
    assert updated is False
    with pytest.warns(UserWarning):
        updated = config_handler._update_new_config(
            {"ocr_model": 21}, {"ocr_model": 20}, parent_key="gutter_detection"
        )
    assert updated is False

    # same value
    updated = config_handler._update_new_config(
        {"jpeg_quality": 95}, {"jpeg_quality": 95}, parent_key="gutter_detection"
    )
    assert updated is False
    updated = config_handler._update_new_config(
        {"gutter_detection": {"fallback_to_center": True}},
        {"gutter_detection": {"fallback_to_center": True}},
        parent_key=None,
    )
    assert updated is False


def test_update_new_config_updated():
    config = {
        "gutter_detection": {
            "output_dir": "./data/out",
            "smooth_kernel": 101,
        }
    }
    updated = config_handler._update_new_config(
        config, {"gutter_detection": {"output_dir": "data"}}, parent_key=None
    )
    assert updated is True
    assert config.get("gutter_detection").get("output_dir") == "data"

    updated = config_handler._update_new_config(
        config["gutter_detection"],
        {"smooth_kernel": 201},
        parent_key="gutter_detection",
    )
    assert updated is True
    assert config.get("gutter_detection").get("smooth_kernel") == 201


def test_save_config_to_file(tmpdir):
    config = {"input_dir": "data"}

    # none dir path
    config_handler.save_config_to_file(config)
    saved_files = get_files(Path.cwd(), "updated_config")
    assert len(saved_files) == 1
    with open(saved_files[0], "r", encoding="utf-8") as f:
        updated_config = json.load(f)
    assert updated_config.get("input_dir") == "data"
    saved_files[0].unlink()  # remove the file

    # valid dir path
    directory = Path(tmpdir.mkdir("test"))
    file_name = "config"
    config_handler.save_config_to_file(config, directory, file_name=file_name + ".json")
    saved_files = get_files(directory, file_name)
    assert len(saved_files) == 1
    with open(saved_files[0], "r", encoding="utf-8") as f:
        updated_config = json.load(f)
    assert updated_config.get("input_dir") == "data"

    # invalid dir path
    file_path = Path(__file__).absolute()
    with pytest.raises(ValueError):
        config_handler.save_config_to_file(config, file_path)


def test_load_config_default():
    config, fname = config_handler.load_config()
    assert config.get("input_dir") == "./data/in"
    assert fname == "default_config"


def test_load_config_file(tmp_path):
    config_path = tmp_path / "config.json"

    # invalid cases
    # not existing file
    with pytest.warns(UserWarning):
        config, fname = config_handler.load_config(config_path, new_config=None)
    assert config.get("input_dir") == "./data/in"  # default config
    assert fname == "default_config"

    # empty file
    open(config_path, "w", newline="", encoding="utf-8").close()
    with pytest.warns(UserWarning):
        config, _ = config_handler.load_config(config_path, new_config=None)
    assert config.get("input_dir") == "./data/in"

    # invalid json file
    with open(config_path, "w", newline="", encoding="utf-8") as f:
        f.write("test")
    with pytest.warns(UserWarning):
        config, fname = config_handler.load_config(config_path, new_config=None)
    assert config.get("gutter_detection").get("number_breakpoints") == 4
    assert fname == "default_config"

    # invalid json file against the schema
    with open(config_path, "w", newline="", encoding="utf-8") as f:
        json.dump({"test": "test"}, f)
    with pytest.warns(UserWarning):
        config, _ = config_handler.load_config(config_path)
    assert config.get("input_dir") == "./data/in"

    # valid json file
    with open(config_path, "w", newline="", encoding="utf-8") as f:
        json.dump({"input_dir": "data"}, f)
    config, fname = config_handler.load_config(config_path=config_path, new_config=None)
    assert config.get("input_dir") == "data"
    assert fname == "config"


def test_load_config_new_config(tmp_path):
    new_config = {"input_dir": "data"}

    # update default config
    config, _ = config_handler.load_config(new_config=new_config)
    assert config.get("input_dir") == "data"

    # update config from file
    config_path = tmp_path / "config.json"
    with open(config_path, "w", newline="", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": "data/in",
            },
            f,
        )
    config, _ = config_handler.load_config(config_path=config_path)
    assert config.get("input_dir") == "data/in"
    config, _ = config_handler.load_config(
        config_path=config_path, new_config=new_config
    )
    assert config.get("input_dir") == "data"

    # update config from file with invalid new config
    new_config = {"test": "test"}
    with pytest.warns(UserWarning):
        config, _ = config_handler.load_config(
            config_path=config_path, new_config=new_config
        )
    assert config.get("input_dir") == "data/in"
