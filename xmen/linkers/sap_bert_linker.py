from itertools import groupby
from typing import List, Union
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

from xmen.linkers import EntityLinker
from xmen.ext.sapbert.src.model_wrapper import Model_Wrapper

from xmen.linkers.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer

from xmen.log import logger

from scipy.spatial.distance import cosine

_SAP_BERT_XLMR = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
_SAP_BERT_EN = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

_EMBED_DIM = 768


class SapBERTLinker(EntityLinker):
    """
    A class that performs entity linking using the SapBERT model.

    Args:
    - index_base_path (Union[Path, str]): Path to the directory where the index files are located.
    - kb_name (str): The name of the knowledge base to use.
    - cuda (bool): Whether to use the GPU for inference or not.
    - gpu_batch_size (int): Batch size for inference on GPU.
    - k (int): The number of closest neighbors to retrieve from the index.
    - threshold (float): A threshold value for filtering out candidate entities based on their similarity score.
    - filter_types (bool): Whether to filter out candidate entities based on their semantic type.
    - embedding_model_name (str): Name of the embedding model to use.
    - consider_n_grams (list): A list of integers representing the number of grams to consider when generating candidate entities.
    - remove_duplicates (bool): Whether to remove duplicate candidate entities or not.
    - expand_abbreviations (bool): Whether to expand abbreviations or not.
    - approximate (bool): Whether to use the hierarchical index for faster inference or not.
    - unique_aliases_only (bool): Whether to use only unique aliases of entities for linking or not.
    """

    # Global state
    instance = None
    model_wrapper = None
    candidate_dense_embeds = None
    term_dict = None

    # Ignore caching when using dataset.map
    def __getstate__(self):
        return {}

    @staticmethod
    def clear():
        if SapBERTLinker.instance:
            SapBERTLinker.instance.valid = False
        SapBERTLinker.instance = None
        SapBERTLinker.model_wrapper = None
        SapBERTLinker.candidate_dense_embeds = None
        SapBERTLinker.term_dict = None

    @staticmethod
    def write_index(
        index_base_path: Union[str, Path],
        term_dict: dict,
        embedding_model_name: str = _SAP_BERT_XLMR,
        cuda: bool = True,
        subtract_mean=True,
        batch_size=2048 * 6,
        write_flat=False,
    ):
        """
        Write index files to disk.

        Args:
        - index_base_path (Union[str, Path]): Path to the directory where the index files will be written.
        - term_dict (dict): A dictionary containing the terms to include in the index.
        - embedding_model_name (str): Name of the embedding model to use.
        - cuda (bool): Whether to use the GPU for inference or not.
        - subtract_mean (bool): Whether to subtract the mean from the embeddings or not.
        - batch_size (int): Batch size for embedding generation.
        - write_flat (bool): Whether to write the flat index to disk or not.
        """
        index_base_path = Path(index_base_path)

        index_base_path.mkdir(exist_ok=True, parents=True)

        out_dict_file = index_base_path / "dict.pickle"
        out_faiss_flat_file = index_base_path / "embed_faiss_flat.pickle"
        out_faiss_hier_file = index_base_path / "embed_faiss_hier.pickle"

        term_dict.to_pickle(out_dict_file)

        wrapper = Model_Wrapper()
        wrapper.load_model(embedding_model_name, use_cuda=cuda)

        logger.info(f"Computing dictionary embeddings with {embedding_model_name}")
        candidate_dense_embeds = wrapper.embed_dense(
            term_dict.term.tolist(),
            agg_mode="cls",
            use_cuda=cuda,
            show_progress=True,
            batch_size=batch_size,
        )
        if subtract_mean:
            candidate_dense_embeds -= candidate_dense_embeds.mean(0)
        # print('Writing embeddings to', out_embed_file)
        # with open(out_embed_file, "wb") as f:
        #    pickle.dump(candidate_dense_embeds, f)

        logger.info("Building FAISS Hierarchical Index")
        hier_indexer = DenseHNSWFlatIndexer(candidate_dense_embeds.shape[1])
        hier_indexer.index_data(candidate_dense_embeds, show_progress=True)
        logger.info(f"Writing FAISS Hierarchical index to {out_faiss_hier_file}")
        hier_indexer.serialize(str(out_faiss_hier_file))

        if write_flat:
            logger.info("Building FAISS Flat Index")
            flat_indexer = DenseFlatIndexer(candidate_dense_embeds.shape[1])
            flat_indexer.index_data(candidate_dense_embeds, show_progress=True)
            flat_indexer.serialize(str(out_faiss_flat_file))

    def __init__(
        self,
        index_base_path: Union[Path, str],
        kb_name: str = "UMLS",
        cuda=True,
        gpu_batch_size: int = 16,
        k: int = 10,
        threshold: float = 0.0,
        filter_types: bool = False,
        embedding_model_name: str = _SAP_BERT_XLMR,
        consider_n_grams: list = [],
        remove_duplicates=True,
        expand_abbreviations=False,
        approximate=True,
        unique_aliases_only=False,
    ):
        index_base_path = Path(index_base_path)
        term_dict_pkl = index_base_path / "dict.pickle"

        if SapBERTLinker.instance:
            raise Exception("SapBERTLinker is a singleton")
        SapBERTLinker.model_wrapper = Model_Wrapper()
        SapBERTLinker.model_wrapper.load_model(embedding_model_name, use_cuda=cuda)
        self.cuda = cuda
        if approximate:
            logger.info("Loading hierarchical faiss index")
            self.indexer = DenseHNSWFlatIndexer(_EMBED_DIM)
            self.indexer.deserialize_from(str(index_base_path / "embed_faiss_hier.pickle"))
        else:
            logger.info("Loading flat faiss index")
            self.indexer = DenseFlatIndexer(_EMBED_DIM)
            self.indexer.deserialize_from(str(index_base_path / "embed_faiss_flat.pickle"))
        # with open(dict_embeddings_pkl, "rb") as f:
        #    get_logger().info("Loading embeddings")
        #    SapBERTLinker.candidate_dense_embeds = pickle.load(f)
        with open(term_dict_pkl, "rb") as f:
            SapBERTLinker.term_dict = pd.read_pickle(f)
            self.term_dict_idx = SapBERTLinker.term_dict.set_index("cui")
        self.k = k
        self.kb_name = kb_name
        self.threshold = threshold
        self.filter_types = filter_types
        self.consider_n_grams = consider_n_grams
        self.filter_types = filter_types
        self.gpu_batch_size = gpu_batch_size
        self.remove_duplicates = remove_duplicates
        self.expand_abbreviations = expand_abbreviations
        self.unique_aliases_only = unique_aliases_only

        SapBERTLinker.instance = self
        self.valid = True

    def predict_batch(self, dataset, batch_size):
        """
        Perform entity linking on a batch of sentences.

        Args:
        - dataset (Dataset): A `Dataset` object containing the sentences to link entities in.
        - batch_size (int): The batch size to use for linking.

        Returns:
        - List of lists of tuples: A list of lists of tuples. Each tuple contains information about a linked entity,
        - including its URI, label, score, and span in the sentence.

        Raises:
        - Exception: If `SapBERTLinker` instance has been cleared, the linker instance is in an inconsistent state.
        """
        if not self.valid:
            raise Exception("SapBERT instance has been cleared, this linker instance is in an inconsistent state.")
        expand_abbreviations = self.expand_abbreviations

        def get_result(sample):
            def get_str(mention):
                s = " ".join(mention["text"])
                if expand_abbreviations and mention.get("long_form", None):
                    s += " [SEP] " + mention["long_form"]
                return s

            mentions_index, mention_strings = zip(
                *[
                    (j, get_str(mention))
                    for j, doc_entities in enumerate(sample["entities"])
                    for mention in doc_entities
                ]
            )

            concepts = self._get_concepts(list(mention_strings))

            entities = sample["entities"]

            for mi, concept_group in groupby(zip(mentions_index, concepts), key=lambda p: p[0]):
                for j, c in enumerate(concept_group):
                    entities[mi][j]["normalized"] = c[1]

            return {"entities": entities}

        return dataset.map(
            function=get_result,
            batch_size=batch_size,
            batched=True,
            load_from_cache_file=False,
        )

    def predict(self, unit: str, entities: dict) -> dict:
        raise NotImplementedError()

    def _get_concepts(self, mention_candidates):

        logger.debug(f"Calculate embeddings for {len(mention_candidates)} mentions")
        mention_dense_embeds = SapBERTLinker.model_wrapper.embed_dense(
            mention_candidates,
            show_progress=False,
            agg_mode="cls",
            use_cuda=self.cuda,
            batch_size=self.gpu_batch_size,
        )
        scores, candidate_idxs_batch, vectors = self.indexer.search_knn(mention_dense_embeds, self.k)

        for i, embed in enumerate(mention_dense_embeds):
            candidates = candidate_idxs_batch[i]
            top_scores = scores[i]
            vector = vectors[i]
            top_scores = [1 - cosine(embed, v[0:_EMBED_DIM]) for v in vector]
            concepts = []
            cuis = set()
            for r, score in zip(SapBERTLinker.term_dict.iloc[candidates].iterrows(), top_scores):
                r = r[1]
                if score > self.threshold:
                    if not self.remove_duplicates or not r.cui in cuis:
                        cuis.add(r.cui)
                        concepts.append((r.cui, score))
            yield [
                {"db_id": cui, "score": score, "db_name": self.kb_name}
                for cui, score in sorted(concepts, key=lambda p: -p[1])
            ]
