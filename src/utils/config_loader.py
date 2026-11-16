"""
CanineVision AI — YAML Configuration Loader.

Why this matters:
    Hard-coded values scattered through code create maintenance nightmares.
    When you want to change the batch size from 32 to 16, you should change
    ONE line in a config file — not hunt through 10 Python files.

    This module provides a single function that reads YAML files and returns
    a plain Python dictionary. Every other module imports its settings
    from here rather than defining magic numbers inline.

Usage:
    from src.utils.config_loader import load_config
    cfg = load_config("configs/config.yaml")
    dataset_root = cfg["paths"]["dataset_root"]
"""

from pathlib import Path

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str | Path) -> dict:
    """
    Load a YAML configuration file and return it as a dictionary.

    Args:
        config_path: Path to the YAML file, relative or absolute.

    Returns:
        Dictionary containing all configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path.resolve()}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.debug("Loaded config: %s", config_path.resolve())
    return config


def load_all_configs(configs_dir: str | Path = "configs") -> dict:
    """
    Load all YAML config files in the configs directory and merge them
    into a single dictionary keyed by filename stem.

    Example:
        configs/config.yaml        → result["config"]
        configs/model_config.yaml  → result["model_config"]
        configs/training_config.yaml → result["training_config"]

    Args:
        configs_dir: Path to the directory containing YAML files.

    Returns:
        Merged dictionary of all configs.
    """
    configs_dir = Path(configs_dir)
    merged: dict = {}

    yaml_files = sorted(configs_dir.glob("*.yaml"))
    if not yaml_files:
        logger.warning("No YAML config files found in: %s", configs_dir.resolve())
        return merged

    for yaml_file in yaml_files:
        cfg = load_config(yaml_file)
        merged[yaml_file.stem] = cfg
        logger.debug("Loaded config file: %s", yaml_file.name)

    logger.info("Loaded %d config file(s) from: %s", len(yaml_files), configs_dir)
    return merged
