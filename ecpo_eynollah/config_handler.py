from pathlib import Path
from importlib import resources
import json
import jsonschema
import warnings
from typing import Dict, Any, Tuple
from typing import Optional


pkg = resources.files("ecpo_eynollah")
DEFAULT_CONFIG_FILE = Path(pkg / "config" / "default_config.json")
CONFIG_SCHEMA_FILE = Path(pkg / "config" / "config_schema.json")


def is_non_empty_file(file_path: Path) -> bool:
    """Check if a file exists and is not empty.

    Args:
        file_path (Path): The path to the file.

    Returns:
        bool: True if the file exists and is not empty, False otherwise.
    """
    invalid_file = (
        not file_path or not file_path.exists() or file_path.stat().st_size == 0
    )
    if invalid_file:
        return False

    return True


def is_valid_config(config: Dict[str, Any], parent_key: str | None) -> bool:
    """Check if the config under parent_key is valid according to the schema.
    Args:
        config (Dict[str, Any]): The configuration.
        parent_key (str | None): The parent key of the config to validate.
            None means the whole config.

    Returns:
        bool: True if the config is valid, False otherwise.
    """
    config_schema = json.load(open(CONFIG_SCHEMA_FILE, "r", encoding="utf-8"))

    if parent_key is not None:
        config_schema = config_schema["properties"].get(parent_key, {})

    try:
        jsonschema.validate(instance=config, schema=config_schema)
        return True
    except jsonschema.ValidationError as e:
        print(e)
        return False


def _update_new_config(
    config: Dict[str, Any], new_config: Dict[str, Any], parent_key: str | None = None
) -> bool:
    """Update the cnfig directly with the new config.

    Args:
        config (Dict[str, Any]): The config.
        new_config (Dict[str, Any]): The new config.
        parent_key (str | None): The parent key of the config to update.
            None means the whole config. Defaults to None.

    Returns:
        bool: True if the config are updated, False otherwise.
    """
    updated = False
    if not config:
        raise ValueError("Current config are empty")

    for key, new_value in new_config.items():
        # check if the new value is different from the old value
        if isinstance(new_value, dict) and key in config:
            # recursively update the nested config
            nested_updated = _update_new_config(config[key], new_value, parent_key=key)
            if nested_updated:
                updated = True
            continue

        updatable = key in config and config[key] != new_value
        if key not in config:
            warnings.warn(
                "Key {} not found in the config and will be skipped.".format(key),
                UserWarning,
            )
        if updatable:
            old_value = config[key]
            config[key] = new_value
            if is_valid_config(config, parent_key=parent_key):
                updated = True
            else:
                warnings.warn(
                    "The new value for key {} is not valid in the config. "
                    "Reverting to the old value: {}".format(key, old_value),
                    UserWarning,
                )
                config[key] = old_value

    return updated


def save_config_to_file(
    config: Dict[str, Any],
    dir_path: Optional[str] = None,
    file_name: str = "updated_config.json",
):
    """Save the config to a file.
    If dir_path is None, save to the current directory.

    Args:
        config (Dict[str, Any]): The config.
        dir_path (str, optional): The path to save the config file.
            Defaults to None.
        file_name (str, optional): The name for the config file.
            Defaults to "updated_config.json".

    Raises:
        ValueError: If the dir_path exists and is not a directory.
    """
    file_path = ""

    if dir_path is None:
        file_path = Path.cwd() / file_name
    else:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            file_path = Path(dir_path) / file_name
        except FileExistsError:
            raise ValueError(
                "The path {} already exists and is not a directory".format(dir_path)
            )

    # save the config to a file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print("The config have been saved to {}".format(file_path))


def load_config(
    config_path: Path | str = "default",
    new_config: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], str]:
    """Get the configuration.
    If the config path is "default", return the default configuration.
    If the config path is not default, read the config from the file.
    If the new config are provided, overwrite the default/loaded config.

    Args:
        config_path (Path | str): Path to the config file.
            Defaults to "default".
        new_config (Dict[str, Any] | None): New config to overwrite the existing config.
            Defaults to {}.

    Returns:
        Tuple[Dict[str, Any], str]: A tuple containing the config dictionary
            and the name of the config file.
    """
    config = {}
    config_fname = ""
    default_setting_path = DEFAULT_CONFIG_FILE

    if not default_setting_path or not is_non_empty_file(default_setting_path):
        raise ValueError(f"Default config file not found or is empty.")

    def load_json(file_path: Path) -> Tuple[Dict[str, Any], str]:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file), file_path.stem

    try:
        config, config_fname = (
            load_json(default_setting_path)
            if config_path == "default"
            else load_json(Path(config_path))
        )
        if config_path != "default" and not is_valid_config(config, parent_key=None):
            warnings.warn(
                "Invalid config file. Using default config instead.",
                UserWarning,
            )
            config, config_fname = load_json(default_setting_path)
    except Exception:
        warnings.warn(
            "Error in loading the config file. Using default config instead.",
            UserWarning,
        )
        config, config_fname = load_json(default_setting_path)

    # update the config with the new config
    if new_config and isinstance(new_config, dict):
        _update_new_config(config, new_config, parent_key=None)

    return config, config_fname
