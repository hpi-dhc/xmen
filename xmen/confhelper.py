from pathlib import Path
from omegaconf import OmegaConf


def load_config(file_name):
    """
    Loads and returns the configuration stored in a YAML file using OmegaConf.

    Args:
    - file_name (str): The path to the YAML file containing the configuration.

    Returns:
    - An OmegaConf object representing the YAML file's contents.

    Note:
        If the configuration file has a `base_config` attribute, this function will attempt to load and merge the base
        configuration file with the current configuration file. The base configuration file should be a YAML file as well,
        and its path is relative to the current configuration file's directory.

    """
    path = Path(file_name)
    conf = OmegaConf.load(path)
    if base_config_path := conf.get("base_config", None):
        base_conf = OmegaConf.load(path.parent / base_config_path)
        conf = OmegaConf.merge(base_conf, conf)
    return conf
