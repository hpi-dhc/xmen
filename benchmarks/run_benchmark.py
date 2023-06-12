import argparse
from pathlib import Path

from xmen.confhelper import load_config
from xmen.log import logger

from dataloaders import load_dataset

def main(config_name) -> None:
    """Run a benchmark with the given config file."""
    
    config = load_config(config_name)
    base_path = Path(config.work_dir)
    
    dict_name = base_path / f"{config.name}.jsonl"

    if not dict_name.exists():
        logger.error(f"{dict_name} does not exist, please run: xmen dict {config_name}")
        return

    index_base_path = base_path / 'index'

    if not index_base_path.exists():
        logger.error(f"{index_base_path} does not exist, please run: xmen index {config_name} --all")
        return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("config", type=str)
    args = argparser.parse_args()

    main(args.config)
