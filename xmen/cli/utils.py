import os
import logging
import typer
from ..log import logger
from pathlib import Path
from ..confhelper import load_config

cfg_path_help = "Path to the .yaml file specifying the configuration."
code_help = "Path to the python script to parse a non-umls kwnoledge base."
output_help = (
    "Path to the directory where the files will be generated. If not provided, the default will be "
    "~/.cache/xmen/{kb_name}."
)

log_file_help = (
    "Path to store the log output locally. Multiple commands to the same log will be "
    "appended, not overwritten. If not provided, no log will be stored."
)

dict_dir_help = (
    "Custom path to the .jsonl dict file from which the indicess will be generated. Provide only if such "
    ".jsonl is not stored in {work_dir} in the config .yaml file."
)

overwrite_help = (
    "If set to True, command will by default overwrite dicts and indexes when encountered with the same "
    "output directories and files, instead of asking if the user wishes to overwrite for each case."
)

custom_name_help = (
    "Provide a name for the dictionary to be built other than the name specified in the config .yaml"
    ". Use for datasets with multiple knowledge bases. The file extension is added automatically."
)

key_help = "Provide the name of the key in the .yaml config file that contains the desired subconfig for" " the dict."

sapbert_help = "Build SapBERT indices."
ngram_help = "Build N-Gram indices."
all_help = "Build all available indicess"
gpu_id_help = "Indicate in which cuda GPU device to generate the indices. Use `-1` to run on the CPU instead."


def get_work_dir(cfg) -> Path:
    """
    Retrieves the working directory specified in the configuration or creates a default directory.

    Args:
    - cfg: a dictionary containing configuration parameters

    Returns:
    - Path object representing the working directory
    """
    if "work_dir" in cfg:
        return Path(cfg.work_dir)
    else:  # default build path
        return Path(os.path.expanduser("~")) / ".cache/xmen" / f"{cfg.name}"


def add_file_logger(log_file):
    """
    Adds a file handler to the logger to save log output to a file.

    Args:
    - log_file: a string representing the path to the log file

    Returns:
    - None
    """
    file_handler = logging.FileHandler(Path(log_file))
    file_handler.setLevel(logging.DEBUG)
    fmt_file = "%(asctime)s - %(levelname)s - %(message)s"
    file_formatter = logging.Formatter(fmt_file)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    pass


def check_and_load_config(cfg_path):
    """
    Warns if the provided path to the configuration file is invalid, otherwise loads it as a DictConf object.

    Args:
    - cfg_path (str): Path to the configuration file to be loaded.

    Returns:
    - (DictConf): Loaded configuration file as a DictConf object.

    Raises:
    - typer.Exit: If the provided path does not exist or is not in the .yaml format.
    """
    if not os.path.exists(cfg_path) or not cfg_path.endswith(".yaml"):
        logger.error(f"File {cfg_path} does not exist or is not in the supported .yaml format.")
        raise typer.Exit()
    else:
        return load_config(cfg_path)


def can_write(path: str, parent_overwrite: bool) -> bool:
    """
    Checks whether the files to be built (dicts or indices) already exist. If so, gives the chance to overwrite them in case it is not prohibited by parent_overwrite flag. the function prompts the user to input either 'y' or 'n' if the file already exists.

    Args:
    - file_path (str): Path to the file being checked.
    - parent_overwrite (bool): Whether the parent command has the `--overwrite` flag set to True.

    Returns:
    - (bool): True if the file can be overwritten, False otherwise.
    """
    if parent_overwrite:
        return True
    else:
        # Handle differently file and directory paths (for dict and index, respectively)
        if "." in str(path).split("/")[-1]:  # check if it is a file, os.path.isfile() works only if the path exists
            is_warnable = os.path.exists(path)
            warning = f"There already exists a file in {path}."
        else:
            is_warnable = os.path.exists(path) and os.listdir(path)
            warning = f"There are already some files in {path}. This command may overwrite them."

        if is_warnable:
            logger.warn(warning)
            logger.info(
                f"To instruct the command to overwrite all coinciding files by default, swith the flag --overwrite."
                f"You can also provide a different path to write the files with --output. \n"
                f"What would you like to do this time? Input yes [y] to overwrite it, no [n] to skip the file."
            )
            overwrite = input("[y/n]")
        else:
            overwrite = "y"

        if overwrite not in ["n", "y"]:
            logger.warn("Unhandled value. Please type `y` for yes or `n` for no.")
            overwrite = can_write(path, parent_overwrite)

        return True if overwrite == "y" else False


def has_correct_keys(cfg, proper_nested_keys: str) -> bool:
    """
    Checks whether the provided configuration dictionary has the specified nested keys.

    Args:
    - cfg (dict): The configuration dictionary to check.
    - proper_nested_keys (str): The required nested keys in the configuration, separated by dots.

    Returns:
    - (bool): True if the dictionary has the required keys, False otherwise.
    """
    nested_keys = proper_nested_keys.split(".")
    for key in nested_keys:
        if key not in cfg:
            logger.error(
                f"The command requires the key `{proper_nested_keys}` to be set in the config .yaml file (in that "
                "specific nested structure)."
            )
            return False
        cfg = cfg[key]
    logger.info(f"Key {proper_nested_keys} correctly found in .yaml file.")
    return True


def get_alias_count(concept_details):
    """
    Calculates the total number of aliases in the given concept_details dictionary.

    Args:
    - concept_details (dict): A dictionary containing details of concepts and their aliases.

    Returns:
    - (int): Total number of aliases, including the original concept name itself.
    """
    return sum([len(c["aliases"]) + 1 for c in concept_details.values()])
