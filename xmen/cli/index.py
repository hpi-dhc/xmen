from omegaconf import DictConfig
from ..log import logger
from xmen.kb import create_flat_term_dict
from xmen.linkers import TFIDFNGramLinker, SapBERTLinker
from pathlib import Path
import torch


def build_ngram(cfg: DictConfig, work_dir: Path, dict_dir: Path):
    """Build an N-Gram index for the given dictionary directory and store it in the working directory.

    Args:
    - cfg (DictConfig): The configuration for building the index.
    - work_dir (Path): The working directory where the index will be stored.
    - dict_dir (Path): The directory containing the dictionary file.
    """

    # ensure the index folder exists
    work_dir.parent.mkdir(exist_ok=True, parents=True)

    # N-Gram indices
    TFIDFNGramLinker.write_index(work_dir / "index" / "ngrams", [dict_dir])
    pass


def build_sapbert(cfg: DictConfig, work_dir: Path, dict_dir: Path, gpu_id: int):
    """Builds an index of concept embeddings using SapBERT.

    Args:
    - cfg (DictConfig): Hydra configuration object.
    - work_dir (Path): Path to the working directory.
    - dict_dir (Path): Path to the directory containing the concept dictionaries.
    - gpu_id (int): ID of the GPU to be used for training. Use -1 for CPU.
    """

    # ensure the index folder exists
    work_dir.parent.mkdir(exist_ok=True, parents=True)
    # SapBERT indices
    term_dict = create_flat_term_dict([dict_dir])
    logger.info(f"Number of aliases: {len(term_dict)}.")
    logger.info(f"Number of concepts: {len(term_dict.cui.unique())}.")

    cuda = False if gpu_id == -1 else True
    with torch.cuda.device(gpu_id):
        SapBERTLinker.write_index(
            work_dir / "index" / "sapbert",
            term_dict=term_dict,
            cuda=cuda,
            subtract_mean=False,
            embedding_model_name=cfg.linker.candidate_generation.sapbert.embedding_model_name,
        )
