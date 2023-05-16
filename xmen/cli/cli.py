import typer
import os
import logging
from pathlib import Path

from .dict import *
from .index import *
from .utils import *
from ..log import logger


app = typer.Typer()


@app.command("dict")
def assemble_kb(
    cfg_path: str = typer.Argument(..., help=cfg_path_help, show_default=False),
    code: str = typer.Option(default=None, help=code_help, show_default=True),
    output: str = typer.Option(default=None, help=output_help, show_default=True),
    key: str = typer.Option(default=None, help=key_help, show_default=True),
    log_file: str = typer.Option(default=None, help=log_file_help, show_default=True),
    overwrite: bool = typer.Option(default=False, help=overwrite_help, show_default=True),
    custom_name: str = typer.Option(default=None, help=custom_name_help, show_default=True),
):
    """
    Builds a .jsonl dictionary with the given configuration for UMLS or custom knowledge bases.

    Args:
    - cfg_path (str): Path to the configuration file.
    - code (Optional[str]): Path to the custom parsing script.
    - output (Optional[str]): Path to the output directory.
    - key (Optional[str]): Name of the subconfig.
    - log_file (Optional[str]): Path to the log file.
    - overwrite (bool): Flag indicating whether to overwrite the existing file in the output directory.
    - custom_name (Optional[str]): Name of the custom knowledge base.

    Raises:
    - typer.Exit: If the provided config file does not contain `umls` nor `custom` keys nested within `dict`,
            and no key is provided with --key.
    """
    if log_file is not None:
        add_file_logger(log_file)
    logger.debug("\n \n NEW DICT RUN")
    cfg = check_and_load_config(cfg_path)

    # make sure there is umls or custom or subconfig key in dict
    if "umls" not in cfg.dict and "custom" not in cfg.dict and not key:
        logger.error(
            "The provided config file does not contain `umls` nor `custom` keys nested within `dict`."
            "If you are attempting to use one of multiple subconfigs, provided it's key name with --key."
        )
        raise typer.Exit()

    # if a subconfig key is given, prune the rest of them out
    if key:
        cfg.dict = cfg.dict[key]
        cfg.name = key

    # custom_name prevails over the key name of a subconfig
    name = custom_name if custom_name is not None else cfg.name
    dict_dir = get_work_dir(cfg) / f"{name}.jsonl" if output is None else Path(output) / f"{name}.jsonl"

    if can_write(dict_dir, overwrite):
        # handle umls kbs
        if "umls" in cfg.dict:
            logger.info(f"Attempting to build {os.path.basename(cfg_path)} in {dict_dir}. This can take some minutes.")
            assemble(cfg, output=dict_dir)
            logger.info(f"Success! {os.path.basename(cfg_path)} assembled in {dict_dir}")
        # handle non-umls without (or wrong) custom code provided
        elif code is None:
            logger.error(
                f"No custom code provided for non-UMLS kb {name}. To use custom code to parse {name}, use flag"
                f" `--code path-to-script/script.py`. If {name} uses UMLS, the problem may be in the provided "
                f".yaml file: the expected file should have key `umls` nested within key `dict`"
            )
        # handle non-umls kbs with correct custom parsing script
        elif code is not None or "custom" in cfg.dict:
            logger.info(
                f"Attempting to build {os.path.basename(cfg_path)} from custom script in {code} to {dict_dir}. This "
                "can take some minutes."
            )
            assemble(cfg, output=dict_dir, custom_path=code)
            logger.info(f"Success! {os.path.basename(cfg_path)} assembled in {dict_dir}")
    else:
        logger.info("Exiting app without building knowledge base dictionary.")
        raise typer.Exit()

    pass


@app.command("index")
def build_ngram_sapbert(
    cfg_path: str = typer.Argument(..., help=cfg_path_help, show_default=False),
    gpu_id: int = typer.Option(default=0, help=gpu_id_help, show_default=True),
    output: str = typer.Option(default=None, help=output_help, show_default=True),
    log_file: str = typer.Option(default=None, help=log_file_help, show_default=True),
    dict: str = typer.Option(default=None, help=dict_dir_help, show_default=True),
    overwrite: bool = typer.Option(default=False, help=overwrite_help, show_default=True),
    sapbert: bool = typer.Option(default=False, help=sapbert_help, show_default=True),
    ngram: bool = typer.Option(default=False, help=ngram_help, show_default=True),
    all: bool = typer.Option(default=False, help=all_help, show_default=True),
):
    """
    Builds N-Gram and SAPBert indices from a .jsonl dict with the given configuration.

    Args:
    - cfg_path (str): Path to the configuration file.
    - gpu_id (int): Index of the GPU to be used for building the SAPBert indices.
    - output (str): Path to the directory where the indices will be stored. If None, the work directory will be used.
    - log_file (str): Path to the file where logs will be written. If None, logs will be printed to console.
    - dict (str): Path to the .jsonl dictionary. If None, the work directory and configuration file name will be used to search for the dictionary.
    - overwrite (bool): Whether to overwrite existing indices or not.
    - sapbert (bool): Whether to build SAPBert indices or not.
    - ngram (bool): Whether to build N-Gram indices or not.
    - all (bool): Whether to build both SAPBert and N-Gram indices or not.

    Raises:
    - typer.Exit: If neither sapbert nor ngram nor all is set to True.
    """
    if log_file is not None:
        add_file_logger(log_file)
    logger.debug("\n \n NEW INDEX RUN")
    cfg = check_and_load_config(cfg_path)
    work_dir = get_work_dir(cfg)
    output = work_dir if output is None else Path(output)
    dict = work_dir / f"{cfg.name}.jsonl" if dict is None else dict

    if not sapbert and not ngram and not all:
        logger.warn("Please indicate which indicess to build with --sapbert or --ngram or --all.")
        raise typer.Exit()

    if os.path.exists(dict):
        logger.info(f"Dictionary found in {dict}. Attempting to build indices.")

        is_ngram_selected = ngram or all
        required_key = "linker.candidate_generation.ngram"
        write_path = output / "index" / "ngrams"
        if is_ngram_selected and has_correct_keys(cfg, required_key) and can_write(write_path, overwrite):
            logger.info("Building N-Gram indices.")
            build_ngram(cfg, output, dict)
        else:
            logger.info("Skipping N-Gram indices.")

        is_sapbert_selected = sapbert or all
        required_key = "linker.candidate_generation.sapbert"
        write_path = output / "index" / "sapbert"
        if is_sapbert_selected and has_correct_keys(cfg, required_key) and can_write(write_path, overwrite):
            # check for GPUs
            if torch.cuda.is_available():
                logger.info(f"CUDA is available. Running on GPU with ID {gpu_id}. To select another, use --gpu-id.")
            else:
                gpu_id = -1
                logger.warning(
                    "CUDA is not available on this system. Running on CPU. This can take considerably longer."
                )
            logger.info("Building SapBERT indices.")
            build_sapbert(cfg, output, dict, gpu_id)
        else:
            logger.info("Skipping SapBERT indices.")

    else:
        logger.info(
            f"Dictionary {dict} does not exist. Build one with command --dict or provide the right path to one with "
            "--dict-dir flag."
        )

    pass


def main():
    app()
