from omegaconf import DictConfig, OmegaConf
from ..log import logger
from xmen.kb import create_flat_term_dict
from xmen.linkers import TFIDFNGramLinker, SapBERTLinker
from pathlib import Path
import torch

DEFAULT_BATCH_SIZE = 2048
DEFAULT_INDEX_BUFFER_SIZE = 50000


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


def build_sapbert(
    cfg: DictConfig,
    work_dir: Path,
    dict_dir: Path,
    gpu_id: int,
    batch_size: int,
    index_buffer_size: int,
    save_ram: bool,
):
    """Builds an index of concept embeddings using SapBERT.

    Args:
    - cfg (DictConfig): Hydra configuration object.
    - work_dir (Path): Path to the working directory.
    - dict_dir (Path): Path to the directory containing the concept dictionaries.
    - gpu_id (int): ID of the GPU to be used for training. Use -1 for CPU.
    - batch_size (int): GPU batch size
    - index_buffer_size (int): Buffer size for use for writing SapBERT FAISS Index
    """

    # ensure the index folder exists
    work_dir.parent.mkdir(exist_ok=True, parents=True)
    # SapBERT indices
    term_dict = create_flat_term_dict([dict_dir])
    logger.info(f"Number of aliases: {len(term_dict)}.")
    logger.info(f"Number of concepts: {len(term_dict.cui.unique())}.")

    if cfg.get("linker", {}).get("candidate_generation", {}).get("sapbert", {}).get("model_name", {}) == {}:
        model_name = SapBERTLinker.CROSS_LINGUAL  # default model
    else:
        model_name = cfg.linker.candidate_generation.sapbert.model_name

    cuda = False if gpu_id == -1 else True
    with torch.cuda.device(gpu_id):
        SapBERTLinker.write_index(
            work_dir / "index" / "sapbert",
            term_dict=term_dict,
            cuda=cuda,
            subtract_mean=False,
            model_name=model_name,
            batch_size=batch_size,
            index_buffer_size=index_buffer_size,
            write_memory_map=save_ram,
        )
