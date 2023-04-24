import pickle
from pathlib import Path
import joblib
import json
import numpy as np

import warnings

from typing import Union, List

from xmen.linkers import EntityLinker, logger
from xmen.knowledge_base import CompositeKnowledgebase

from scispacy.candidate_generation import (
    CandidateGenerator,
    load_approximate_nearest_neighbours_index,
    LinkerPaths,
    create_tfidf_ann_index,
)
from scispacy.linking import EntityLinker as ScispacyLinker
from scispacy.linking_utils import KnowledgeBase


class TFIDFNGramLinker(EntityLinker):
    @staticmethod
    def write_index(index_base_path: Union[str, Path], jsonl_files: List[Union[str, Path]]):
        kb = CompositeKnowledgebase(jsonl_files)

        index_base_path.mkdir(exist_ok=True, parents=True)

        pickle.dump(kb, open(Path(index_base_path) / "kb.pickle", "wb"))

        create_tfidf_ann_index(index_base_path, kb)

    @staticmethod
    def default_scispacy():
        return TFIDFNGramLinker.scispacy()

    @staticmethod
    def scispacy_no_thresholds():
        return TFIDFNGramLinker.scispacy(
            k=100,
            threshold=0.0,
            filter_for_definitions=False,
        )

    @staticmethod
    def load_candidate_generator(index_base_path: Union[str, Path]):
        index_base_path = Path(index_base_path)
        lp = LinkerPaths(
            ann_index=index_base_path / "nmslib_index.bin",
            tfidf_vectorizer=index_base_path / "tfidf_vectorizer.joblib",
            tfidf_vectors=index_base_path / "tfidf_vectors_sparse.npz",
            concept_aliases_list=index_base_path / "concept_aliases.json",
        )
        ann_index = load_approximate_nearest_neighbours_index(lp)
        tfidf_vectorizer = joblib.load(lp.tfidf_vectorizer)
        ann_concept_aliases_list = json.load(open(lp.concept_aliases_list, encoding="utf-8"))
        kb = pickle.load(open(index_base_path / "kb.pickle", "rb"))

        cg = CandidateGenerator(ann_index, tfidf_vectorizer, ann_concept_aliases_list, kb)
        return cg

    @staticmethod
    def scispacy(
        expand_abbreviations: bool = None,
        k: int = None,
        threshold: float = None,
        no_definition_threshold: float = None,
        filter_for_definitions: bool = None,
    ):
        logger.info("Initializing scispaCy")
        default_linker = ScispacyLinker()
        return TFIDFNGramLinker(
            default_linker.candidate_generator,
            kb=None,
            kb_name="UMLS",
            expand_abbreviations=expand_abbreviations if expand_abbreviations else default_linker.resolve_abbreviations,
            k=k if k else default_linker.k,
            threshold=threshold if threshold else default_linker.threshold,
            no_definition_threshold=no_definition_threshold
            if no_definition_threshold
            else default_linker.no_definition_threshold,
            filter_for_definitions=filter_for_definitions
            if filter_for_definitions
            else default_linker.filter_for_definitions,
            filter_types=False,
        )

    # Ignore caching when using dataset.map
    def __getstate__(self):
        return {}

    def __init__(
        self,
        candidate_generator: CandidateGenerator = None,
        index_base_path: Union[Path, str] = None,
        kb: KnowledgeBase = None,
        kb_name: str = "UMLS",
        expand_abbreviations: bool = True,
        k: int = 30,
        threshold: float = 0.0,
        no_definition_threshold: float = 0.95,
        filter_for_definitions: bool = False,
        filter_types: bool = False,
    ):
        if candidate_generator:
            self.candidate_generator = candidate_generator
        elif index_base_path:
            self.candidate_generator = TFIDFNGramLinker.load_candidate_generator(index_base_path)
        else:
            raise Exception("Either candidate_generator or index_base_path have to be provided")
        self.expand_abbreviations = expand_abbreviations
        self.k = k
        self.kb = kb if kb else self.candidate_generator.kb
        self.kb_name = kb_name
        self.threshold = threshold
        self.no_definition_threshold = no_definition_threshold
        self.filter_for_definitions = filter_for_definitions
        self.filter_types = filter_types

    def predict(self, passages: list, entities: list) -> dict:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.VisibleDeprecationWarning)

            def get_m_string(e):
                if self.expand_abbreviations and e.get("long_form", None):
                    return e["long_form"]
                else:
                    return " ".join(e["text"])

            mention_strings = [get_m_string(e) for e in entities]
            batch_candidates = self.candidate_generator(mention_strings, self.k)

            for _, candidates, e in zip(mention_strings, batch_candidates, entities):
                predicted = []
                for cand in candidates:
                    score = max(cand.similarities)
                    if (
                        self.filter_for_definitions
                        and self.kb.cui_to_entity[cand.concept_id].definition is None
                        and score < self.no_definition_threshold
                    ):
                        continue
                    if score > self.threshold:
                        predicted.append((cand.concept_id, score))
                sorted_predicted = sorted(predicted, reverse=True, key=lambda x: x[1])
                e["normalized"] = [
                    {"db_id": cui, "score": score, "db_name": self.kb_name} for cui, score in sorted_predicted
                ]
        return entities
