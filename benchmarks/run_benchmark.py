import hydra
from pathlib import Path

from xmen.confhelper import load_config
from xmen.log import logger

from dataloaders import load_dataset

@hydra.main(version_base=None, config_path=".", config_name="benchmark.yaml")
def main(config) -> None:
    """Run a benchmark with the given config file."""

    base_path = Path(config.hydra_work_dir)

    dict_name = base_path / f"{config.benchmark.name}.jsonl"

    if not dict_name.exists():
        logger.error(f"{dict_name} does not exist, please run: xmen dict <config name>")
        return

    index_base_path = base_path / 'index'

    if not index_base_path.exists():
        logger.error(f"{index_base_path} does not exist, please run: xmen index <config name> --all")
        return
    
    dataset = load_dataset(config.benchmark.dataset)

    print(dataset)


if __name__ == "__main__":
    main()