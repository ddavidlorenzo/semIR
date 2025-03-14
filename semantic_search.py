"""
This module implements a semantic textual information retrieval system.
"""

from typing import Union
import pandas as pd
import numpy as np
import os
from annoy import AnnoyIndex
from search import Search
from utils import utils


class SemanticSearch(Search):
    """This class implements a semantic textual information retrieval system,
    which allows for advanced features like retrieve and re-rank and Approximate
    Nearest Neighbours search.
    """

    @property
    def annoy(self) -> AnnoyIndex:
        """Getter method for `annoy`

        :return: Current ANNOY index. If none, it attempts to create a new one
         with the optimal configuration (according to the experiments detailed
         in the dissertation document).
        :rtype: AnnoyIndex
        """
        try:
            return self._annoy
        except AttributeError:
            self.logger.info("Loading Annoy Index.")
            self.annoy = self.get_annoy_index(
                self.path_annoy_cache, self.annoy_n_trees, self.annoy_emb_size
            )
        return self._annoy

    @annoy.setter
    def annoy(self, annoy: AnnoyIndex) -> None:
        self._annoy = annoy

    @property
    def path_annoy_cache(self) -> str:
        return self._path_annoy_cache

    @path_annoy_cache.setter
    def path_annoy_cache(self, path_annoy_cache: str) -> None:
        self._path_annoy_cache = path_annoy_cache

    @property
    def annoy_n_trees(self) -> int:
        return self._annoy_n_trees

    @annoy_n_trees.setter
    def annoy_n_trees(self, annoy_n_trees: int) -> None:
        self._annoy_n_trees = annoy_n_trees

    @property
    def annoy_emb_size(self) -> int:
        return self._annoy_emb_size

    @annoy_emb_size.setter
    def annoy_emb_size(self, annoy_emb_size: int) -> None:
        self._annoy_emb_size = annoy_emb_size

    def __init__(
        self,
        corpus: pd.DataFrame,
        path_embs_cache: str = None,
        encoding_strategy: str = "title_overview",
        path_biencoder: str = "paraphrase-distilroberta-base-v2",
        max_seq_length: int = 512,
        path_crossencoder: str = "cross-encoder/stsb-distilroberta-base",
        path_annoy_cache: str = None,
        annoy_n_trees: int = 576,
        annoy_emb_size: int = 768,
    ):
        """Constructor for a `SemanticSearch` instance.

        **Important**: if you are willing to load the embeddings or the ANNOY
        index from disk you do not need to tune their respective specific
        parameters (they will be overlooked).

        :param corpus: Book corpus. It should, at least, have the following columns
         (with the very same names):
         * `gr_book_id`: book unique identifier.
         * `title`: book titles.
         * `authors`: book authors.
         * `overview`: book overview.

        :type corpus: DataFrame
        :param path_embs_cache: Filepath to store the computed embeddings
         or load the embeddings from. Defaults to None.
        :type path_embs_cache: str, optional
        :param encoding_strategy: Encoding strategy to use. The encoding strategy
         must be a string containing the names of the features to include into
         the input of the encoder, each of them separated by an underscore ('_').
         For example, if you were to use the title and the overview as the encoding
         strategy, `encoding_strategy` must be either `title_overview` or `overview_title`.
         Defaults to 'title_overview'.
        :type encoding_strategy: str, optional
        :param path_biencoder: Model `id` of a pretrained
         `SentenceTransformer` hosted inside a model repo on HuggingFace. Defaults
         to 'paraphrase-distilroberta-base-v2'. Defaults to 'paraphrase-distilroberta-base-v2'.
        :type path_biencoder: str, optional
        :param max_seq_length: Property to get the maximal input sequence
         length for the model. Longer inputs will be truncated. Defaults to 512.
        :type max_seq_length: int, optional
        :param path_crossencoder: Model `id` of a pretrained `CrossEncoder` hosted in 
         HuggingFace. Defaults to None.
        :type path_crossencoder: str, optional
        :param path_annoy_cache: Filepath to store an ANNOY index or load
         it from disk. Defaults to None.
        :type path_annoy_cache: str, optional
        :param annoy_n_trees: Number of trees to use in the forest for ANNOY.
         Defaults to 576
        :type annoy_n_trees: int, optional
        :param annoy_emb_size: Size of the embeddings, required to compute
         the index. Defaults to 768
        :type annoy_emb_size: int, optional
        """
        super().__init__(
            corpus,
            encoding_strategy=encoding_strategy,
            path_biencoder=path_biencoder,
            path_embs_cache=path_embs_cache,
            max_seq_length=max_seq_length,
            path_crossencoder=path_crossencoder,
        )

        self.path_annoy_cache = path_annoy_cache
        self.annoy_n_trees = annoy_n_trees
        self.annoy_emb_size = annoy_emb_size
    
    def search(
        self,
        *queries: Union[str,list[str]],
        k: int = 5,
        k_biencoder: int = 20,
        use_annoy: bool = False,
        reranking: bool = False,
    ) -> Union[list[str], str]:
        """Perform semantic search.

        :param query: Textual query.
        :type query: str
        :param k: Number of most relevant documents to retrieve. When using exhaustive
         search, the value of `k` does not affect perfomance. Complexity using ANNOY and
         `k` ~ corpus length will be close to O(n). Defaults to 5
        :type k: int, optional
        :param k_biencoder: If using retrieve and re-rank, number of documents to
         retrieve by the Bi-encoder and fed into the Cross-Encoder. The Cross-Encoder will
         return the `k` most relevant entries. `k_biencoder` must be greater or equal to
         `k`. Defaults to 20.
        :type k_biencoder: int, optional
        :param use_annoy: Use approximate search to reduce search time to approx O(log(n)).
         Defaults to False
        :type use_annoy: bool, optional
        :param reranking: Use retrieve and Re-Rank Pipeline, defaults to False
        :type reranking: bool, optional
        """
        # Results to retrieve by Bi-encoder is `k_biencoder` if
        # using Retrieve & Re-Rank pipeline.
        top_k = k_biencoder if reranking else k
        # Ensure the number of documents is, at most, the number
        # of entries in the corpus.
        top_k = min(top_k, len(self.input_encoder))

        results = []

        for query in queries:
            # Compute the embedding of the query.
            query_embedding = self.biencoder.encode(query, convert_to_tensor=True)
            # If using approximate search,
            if use_annoy:
                # get results using ANNOY python framework.
                indices, scores = self.annoy.get_nns_by_vector(
                    query_embedding, top_k, include_distances=True
                )
                # Normalize scores (Annoy uses sqrt(2-2*cos(u, v)) as angular distance).
                scores = np.array([1 - ((score**2) / 2) for score in scores])
            else:
                # get results using exhaustive search.
                scores, indices = utils.topk_cos_sim(
                    query_embedding, self.embeddings, top_k
                )

            # Retrieve & Re-Rank pipeline.
            if reranking:
                # Now, score all retrieved documents with the Cross-encoder
                cross_inp = [
                    [query, self.input_encoder[hit_idx]] for hit_idx in indices
                ]
                # Perform self-attention over each pair of sentences and
                # obtain a score.
                cross_scores = self.crossencoder.predict(cross_inp)
                # Compute the indices that would sort the scores obtained by
                # the Cross-Encoder.
                order = cross_scores.argsort()
                # `argsort` sorts in increasing order. To get the k elements
                # with the highest score, we first take the last k elements
                # of order, containing the indexes we are looking for, and then
                # reverse the list (decreasing order).
                scores = cross_scores[order][-k:][::-1]
                # Convert `indices` into a NumPy array to reorder elements easier.
                indices = np.array(indices) if use_annoy else indices.detach().numpy()
                # Following the same rationale, `indices` will store the indices
                # of top `k` retrieved documents.
                indices = indices[order][-k:][::-1]

            # Append query results.
            results.append(self._str_query_results(query, scores, indices, k))

        # Print query results.
        return results if len(results) > 1 else results.pop()

    def get_annoy_index(
        self, index_cache_path: str = None, n_trees=576, embedding_size=768
    ):
        """Get ANNOY index. Use this method to:

        * Use a precomputed ANNOY index located in `index_cache_path`.
        * Create a new ANNOY index and store it in `index_cache_path`.
            if `index_cache_path` is `None`, the index will not be stored in disk.
        * Either way, the obtained ANNOY index will be used in future calls
            to `search` and `search_multiple` if approximate search is chosen.
        * Previous ANNOY setup is replaced upon invoking this method.

        **IMPORTANT**: if you are attempting to load an ANNOY index from disk, there
        is no need to tune the remaining parameters (i.e., `n_trees` and
        `embedding_size`)

        :param index_cache_path: Filepath to store the obtained ANNOY index or filepath
         of a precomputed ANNOY index. By default is `None`: a new ANNOY index will be
         created with the indicated parameters.
        :type index_cache_path: str, optional
        :param n_trees: Number of trees to use in the forest for ANNOY, defaults to 576.
        :type n_trees: int, optional
        :param embedding_size: Size of the embeddings, required to compute the index.
         Defaults to 768
        :type embedding_size: int, optional
        """
        # Instantiate AnnoyIndex object with cosine-similarity metric (`angular`)
        # and indicating the size of the embeddings.
        annoy = AnnoyIndex(embedding_size, "angular")
        if index_cache_path and os.path.exists(index_cache_path):
            self.logger.info("Loading ANNOY index from disk")
            annoy.load(index_cache_path)
        else:
            self.logger.info("Creating ANNOY index. This might take a while.")
            # Index each embedding.
            for idx, embedding in enumerate(self.embeddings):
                annoy.add_item(idx, embedding)
            # Build ANNOY index with `n_trees` projection trees.
            annoy.build(n_trees)
            # If provided a path to store the ANNOY index created
            if index_cache_path:
                utils.makedir(index_cache_path, remove_filename=True)
                # attempt to save it.
                annoy.save(index_cache_path)
        # Return ANNOY index.
        return annoy

    def test_annoy_performance(
        self, queries: list, k=5, verbose=True
    ) -> tuple[float, float]:
        """Utility to test the performance of ANNOY, considering the speedup with respect to
        exhaustive search and the recall.

        :param queries: Collection of textual queries.
        :type queries: list or array-like
        :param k: Top k elements to consider in the comparison.
        :type k: int, optional
        """
        import time as t

        # Find the closest k sentences of the corpus for each query sentence based on cosine similarity
        wrong_ids = []
        time_annoy, time_exhaustive = 0.0, 0.0

        # ANNOY index may not be loaded. To exclude the time taken to load
        # an existant or create a new ANNOY index, we load it before starting
        # the test.
        annoy = self.annoy
        top_k = min(k, len(self.input_encoder))
        for query in queries:
            query_embedding = self.biencoder.encode(query, convert_to_tensor=True)
            tinit = t.time()
            ann_ids = annoy.get_nns_by_vector(
                query_embedding, top_k, include_distances=True
            )[0]
            time_annoy += t.time() - tinit
            ann_ids = set(ann_ids)

            # We use cosine-similarity and torch.topk to find the highest k scores
            tinit = t.time()
            ids = utils.topk_cos_sim(query_embedding, self.embeddings, top_k)[1]
            time_exhaustive += t.time() - tinit
            ids = set([int(id) for id in ids])
            wrong_ids += list(ids - ann_ids)

        # Compute recall
        n_queries = len(queries) * top_k
        recall = (n_queries - len(wrong_ids)) / n_queries
        speedup = time_exhaustive / time_annoy

        if verbose:
            self.logger.info(
                "\n==================================================="
                f"\nApproximate Nearest Neighbor Recall@{top_k}: {recall*100:.2f}%"
                f"\nTotal time using ANNOY {time_annoy:.2f} seconds."
                f"\nTotal time using exhaustive search {time_exhaustive:.2f} seconds."
                f"Speed-up: {speedup:.4f}"
            )

            if recall < 1:
                list_wrong = []
                for i, miss_id in enumerate(wrong_ids):
                    doc = self.corpus.iloc[miss_id]
                    list_wrong.append(
                        f'{i}. "{doc.title}", by {doc.authors} (Goodreads Id:{doc.gr_book_id})'
                    )
                str_wrong = '\n'.join(list_wrong)
                self.logger.debug(
                    "\n==================================================="
                    "\nMissing results:"
                    "\n------------------\n"
                    f"{str_wrong}"
                    "\n==================================================="
                )
        return recall, speedup
