# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
FAISS-based index components. Original from 
https://github.com/facebookresearch/BLINK/blob/main/blink/indexer/faiss_indexer.py
"""

import os
import logging
import pickle

import faiss
import numpy as np

from xmen.log import logger
from tqdm.autonotebook import tqdm


class DenseIndexer(object):
    """
    A class for creating and searching a dense vector index using Faiss.

    Args:
    - buffer_size (int): The maximum number of data points to hold in memory while indexing.

    Attributes:
    - buffer_size (int): The maximum number of data points to hold in memory while indexing.
    - index_id_to_db_id (list): A list that maps the index ID to the original database ID.
    - index (faiss.IndexFlatIP): The Faiss index.
    """

    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def index_data(self, data: np.array, show_progress=False):
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_docs: int):
        raise NotImplementedError

    def serialize(self, index_file: str):
        """
        Serializes the index to a file.

        Args:
        - index_file (str): The path to the file where the serialized index will be stored.
        """
        logger.info("Serializing index to %s", index_file)
        faiss.write_index(self.index, index_file)

    def deserialize_from(self, index_file: str):
        """
        Deserializes the index from a file.

        Args:
        - index_file (str): The path to the file containing the serialized index.
        """
        logger.info("Loading index from %s", index_file)
        self.index = faiss.read_index(index_file)
        logger.info("Loaded index of type %s and size %d", type(self.index), self.index.ntotal)


# DenseFlatIndexer does exact search
class DenseFlatIndexer(DenseIndexer):
    """
    Represents a dense flat indexer object.

    Args:
    - vector_sz (int): Size of the input vector. Default is 1.
    - buffer_size (int): Size of the buffer to use while indexing data. Default is 50000.

    Attributes:
    - index: An instance of faiss.IndexFlatIP initialized with vector_sz.
    """

    def __init__(self, vector_sz: int = 1, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)
        self.index = faiss.IndexFlatIP(vector_sz)

    def index_data(self, data: np.array, show_progress=False):
        """
        Indexes the given data using the IndexFlatIP index.

        Args:
        - data (np.array): The input data to be indexed.
        - show_progress (bool): If True, shows progress bar while indexing. Default is False.
        """
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        logger.info("Indexing data, this may take a while.")
        cnt = 0
        for i in tqdm(range(0, n, self.buffer_size), disable=not show_progress):
            vectors = [np.reshape(t, (1, -1)) for t in data[i : i + self.buffer_size]]
            vectors = np.concatenate(vectors, axis=0)
            self.index.add(vectors)
            cnt += self.buffer_size

        logger.info("Total data indexed %d", n)

    def search_knn(self, query_vectors, top_k):
        """
        Searches the indexed data for the nearest neighbors of the given query vectors.

        Args:
        - query_vectors (np.array): The query vectors for which nearest neighbors are to be found.
        - top_k (int): The number of nearest neighbors to be returned.

        Returns:
        - scores (np.array): The similarity scores of the nearest neighbors.
        - indexes (np.array): The indexes of the nearest neighbors in the indexed data.
        """
        scores, indexes = self.index.search(query_vectors, top_k)
        return scores, indexes


# DenseHNSWFlatIndexer does approximate search
class DenseHNSWFlatIndexer(DenseIndexer):
    """
    Efficient index for retrieval using HNSWFlat algorithm with L2 similarity.
    This indexer supports conversion from DOT product similarity space to L2 similarity space.
    Default settings are for high accuracy but also high RAM usage.

    Args:
    - vector_sz (int): Size of input vectors.
    - buffer_size (int): Batch size for indexing. Default is 50000.
    - store_n (int): Number of vectors to store per index. Default is 128.
    - ef_search (int): Number of neighbors to consider during search. Default is 256.
    - ef_construction (int): Number of neighbors to consider during index construction. Default is 200.

    Attributes:
    - index: Faiss IndexHNSWFlat object for indexing vectors.
    - phi (float): Auxiliary dimension for conversion of DOT product similarity space to L2 similarity space.
    """

    def __init__(
        self,
        vector_sz: int,
        buffer_size: int = 50000,
        store_n: int = 128,
        ef_search: int = 256,
        ef_construction: int = 200,
    ):
        super(DenseHNSWFlatIndexer, self).__init__(buffer_size=buffer_size)

        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        index = faiss.IndexHNSWFlat(vector_sz + 1, store_n)
        index.hnsw.efSearch = ef_search
        index.hnsw.efConstruction = ef_construction
        self.index = index
        self.phi = 0

    def index_data(self, data: np.array, show_progress=False):
        """
        Indexes the input data into the indexer.

        Args:
        - data (np.array): Input data to be indexed.
        - show_progress (bool): If True, shows a progress bar during indexing. Default is False.
        """
        n = len(data)

        # max norm is required before putting all vectors in the index to convert inner product similarity to L2
        if self.phi > 0:
            raise RuntimeError(
                "DPR HNSWF index needs to index all data at once," "results will be unpredictable otherwise."
            )
        phi = 0
        for i, item in enumerate(data):
            doc_vector = item
            norms = (doc_vector**2).sum()
            phi = max(phi, norms)
        logger.info("HNSWF DotProduct -> L2 space phi={}".format(phi))
        self.phi = 0

        # indexing in batches is beneficial for many faiss index types
        logger.info("Indexing data, this may take a while.")
        cnt = 0
        for i in tqdm(range(0, n, self.buffer_size), disable=not show_progress):
            vectors = [np.reshape(t, (1, -1)) for t in data[i : i + self.buffer_size]]

            norms = [(doc_vector**2).sum() for doc_vector in vectors]
            aux_dims = [np.sqrt(phi - norm) for norm in norms]
            hnsw_vectors = [np.hstack((doc_vector, aux_dims[i].reshape(-1, 1))) for i, doc_vector in enumerate(vectors)]
            hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)

            self.index.add(hnsw_vectors)
            cnt += self.buffer_size

        logger.info("Total data indexed %d" % n)

    def search_knn(self, query_vectors, top_k):
        """
        Searches the index for the k-nearest neighbors of the query vectors.

        Args:
        - query_vectors: A numpy array of shape (n_queries, embedding_dim) containing the query vectors.
        - top_k: An integer representing the number of nearest neighbors to retrieve for each query.

        Returns:
        - scores: A numpy array of shape (n_queries, top_k) containing the distances of the k-nearest neighbors for each query.
        - indexes: A numpy array of shape (n_queries, top_k) containing the indexes of the k-nearest neighbors for each query.
        - vectors: A numpy array of shape (n_queries, top_k, embedding_dim) containing the embeddings of the k-nearest neighbors for each query.
        """
        aux_dim = np.zeros(len(query_vectors), dtype="float32")
        query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
        logger.debug("query_hnsw_vectors %s", query_nhsw_vectors.shape)
        scores, indexes, vectors = self.index.search_and_reconstruct(query_nhsw_vectors, top_k)
        return scores, indexes, vectors

    def deserialize_from(self, file: str):
        """
        Deserializes the index from the given file.

        Args:
        - file: A string representing the path to the file containing the serialized index.
        """
        super(DenseHNSWFlatIndexer, self).deserialize_from(file)
        # to trigger warning on subsequent indexing
        self.phi = 1
