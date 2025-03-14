"""
This module implements a textual literal information retrieval system.
"""

from typing import Literal, Union
import pandas as pd
import numpy as np
import os
import torch
from search import Search
from utils import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords


class TfIdfSearch(Search):
    """This class implements a textual literal information retrieval system,
    based on TF-IDF which allows for advanced features like hybrid search,
    combining literal and dense search (retrieve and re-rank search pipeline).
    """

    @property
    def vectorizer(self) -> TfidfVectorizer:
        return self._vectorizer

    @vectorizer.setter
    def vectorizer(self, vectorizer: TfidfVectorizer) -> None:
        self._vectorizer = vectorizer

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

    @vectors.setter
    def vectors(self, vectors: np.ndarray) -> None:
        self._vectors = vectors

    def __init__(
        self,
        corpus: pd.DataFrame,
        vectors_cache_path: str = None,
        encoding_strategy="title_overview",
        path_biencoder: str = "paraphrase-distilroberta-base-v2",
        path_embs_cache: str = None,
        max_seq_length=512,
        path_crossencoder: str = "cross-encoder/stsb-distilroberta-base",
    ):
        """Constructor for a `TfIdfSearch` instance.

        **Important**: if you are willing to load the embeddings or the vectors
        from disk you do not need to tune their respective specific parameters
        (they will be overlooked).

        :param corpus: Book corpus. It should, at least, have the following columns
         (with the very same names):
         * `gr_book_id`: book unique identifier.
         * `title`: book titles.
         * `authors`: book authors.
         * `overview`: book overview.

        :type corpus: DataFrame
        :param vectors_cache_path: Filepath to store the computed vectors
         or load the vectors from. Defaults to None.
        :type vectors_cache_path: str, optional
        :param encoding_strategy: Encoding strategy to use. The encoding strategy
         must be a string containing the names of the features to include into
         the input of the encoder, each of them separated by an underscore ('_').
         For example, if you were to use the title and the overview as the encoding
         strategy, `encoding_strategy` must be either `title_overview` or `overview_title`.
         Defaults to 'title_overview'.
        :type encoding_strategy: str, optional
        :param path_biencoder: The model `id` of a pretrained
         `SentenceTransformer` hosted inside a model repo on HuggingFace. Defaults
         to 'paraphrase-distilroberta-base-v2'. Defaults to None.
        :type path_biencoder: str, optional
        :param path_embs_cache: Filepath to store the computed embeddings
         or load the embeddings from. Defaults to None.
        :type path_embs_cache: str, optional
        :param max_seq_length: Property to get the maximal input sequence
         length for the model. Longer inputs will be truncated. Defaults to 512.
        :type max_seq_length: int, optional
        :param path_crossencoder: Model `id` of a pretrained `CrossEncoder` hosted in HuggingFace. Defaults to None.
        :type path_crossencoder: str, optional
        """
        super().__init__(
            corpus,
            encoding_strategy=encoding_strategy,
            path_biencoder=path_biencoder,
            path_embs_cache=path_embs_cache,
            max_seq_length=max_seq_length,
            path_crossencoder=path_crossencoder,
        )

        # Get the values for the TF-IDF vectorizer, corpus vectors and
        # the list of inputs to the vectorizer or sentence embedder.
        self.vectorizer, self.vectors = self._get_vectors(vectors_cache_path, encoding_strategy)


    def search(
        self,
        *queries: list[str],
        k=5,
        k_lexical=20,
        reranking_strategy: Literal["crossencoder", "biencoder"] = None,
    ) -> Union[list[str], str]:
        """Perform TF-IDF lexical search.

        :param queries: Textual queries.
        :type queries: list[str]
        :param k: Number of most relevant documents to retrieve.
        :type k: int, optional
        :param k_lexical: If using retrieve and re-rank, number of documents to
         retrieve by lexical search and re-ranked by any re-ranking strategy. Re-Ranker
         will return the `k` most relevant entries. `k_lexical` must be greater or equal
         to `k`. Defaults to 20.
        :type k_lexical: int, optional
        :param reranking_strategy: Re-ranking strategy to use. Defaults to None
        :type reranking_strategy: Literal['crossencoder', 'biencoder'], optional
        :raises ValueError: Raise `ValueError` if `reranking_strategy` takes an ilegal
         value.
        """
        # Results to retrieve by TF-IDF is `k_lexical` if
        # using Retrieve & Re-Rank pipeline.
        top_k = k_lexical if reranking_strategy else k
        # Ensure the number of documents is, at most, the number
        # of entries in the corpus.
        top_k = min(top_k, len(self.input_encoder))

        results = []

        for query in queries:
            # Vectorize the query.
            query_vector = self.vectorizer.transform([query])
            # Compute cosine similarity between query and candidates.
            cos_scores = torch.tensor(cosine_similarity(query_vector, self.vectors))[0]
            # Get the scores and indices of the top k results.
            scores, indices = torch.topk(cos_scores, k=top_k)

            # Retrieve & Re-Rank pipeline.
            if reranking_strategy == "crossencoder" or reranking_strategy == "biencoder":

                if reranking_strategy == "crossencoder":
                    # Now, score all retrieved documents with the Cross-encoder
                    cross_inp = [
                        [query, self.input_encoder[hit_idx]] for hit_idx in indices
                    ]
                    # Perform self-attention over each pair of sentences and
                    # obtain a score.
                    rerank_scores = self.crossencoder.predict(cross_inp)
                else:
                    # Compute the embedding of the query.
                    query_embedding = self.biencoder.encode(query)
                    # Get the embeddings of the `k_lexical` retrieved documents.
                    rerank_embeddings = np.array(
                        [self.embeddings[hit_idx].detach().numpy() for hit_idx in indices]
                    )
                    # Obtain similarity scores.
                    rerank_scores = (
                        utils.topk_cos_sim(query_embedding, rerank_embeddings, k)[0]
                        .detach()
                        .numpy()
                    )
                # Compute the indices that would sort the scores obtained by
                # the Re-Ranker.
                order = rerank_scores.argsort()
                # `argsort` sorts in increasing order. To get the k elements
                # with the highest score, we first take the last k elements
                # of order, containing the indexes we are looking for, and then
                # reverse the list (descreasing order).
                scores = rerank_scores[order][-k:][::-1]
                # Convert `indices` into a NumPy array to reorder elements easier.
                indices = indices.detach().numpy()[order][-k:][::-1]

            elif reranking_strategy:
                # Ilegal value for `reranking_strategy`.
                raise ValueError(
                    f"'{reranking_strategy}' is not a valid re-ranking strategy. "
                    + "Current available options are 'biencoder' and 'crossencoder'"
                )

            # Append query results.
            results.append(self._str_query_results(query, scores, indices, k))

        return results if len(results) > 1 else results.pop()

    def _get_vectors(self, vectors_cache_path: str, encoding_strategy: str, lang:str="english") -> tuple[TfidfVectorizer, np.ndarray]:
        """Load from disk or create:
        * TF-IDF vectorizer fitted with corpus information.
        * Corpus vectorization according to `encoding_stategy`.
        * Collection of inputs to the encoder.

        :param vectors_cache_path: Filepath to store the computed vectors
         or load the vectors from. Defaults to None.
        :type vectors_cache_path: str
        :param encoding_strategy: Encoding strategy to use.
        :type encoding_strategy: str
        :param lang: Corpus language.
        :type lang: str
        :return: Vectorizer object, (pre)computed vectors.
        :rtype: tuple[TfidfVectorizer, np.ndarray]
        """
        # Check whether the vectors' cache path exists
        if vectors_cache_path and os.path.exists(vectors_cache_path):
            self.logger.info("Loading precomputed vectors from disk")
            vectors, input_encoder, vectorizer = utils.load_embeddings(
                vectors_cache_path
            )
        else:
            # Get the list of inputs to the encoder
            input_encoder = utils.prepare_input_encoder(encoding_strategy, self.corpus)
            self.logger.info("Vectorizing input data. This might take a while")
            vectorizer = TfidfVectorizer(stop_words=stopwords.words(lang))
            # Learn vocabulary and idf from training set.
            vectorizer.fit(input_encoder)
            # Transform documents to document-term matrix.
            vectors = vectorizer.transform(input_encoder)
            # If there is a path to save the vectorized documents
            if vectors_cache_path:
                # attempt to store them in disk.
                utils.store_embeddings(
                    vectors,
                    input_encoder=input_encoder,
                    vectorizer=vectorizer,
                    out_filename=vectors_cache_path,
                )

        # Return the values for TF-IDF vectorizer and corpus vectors
        return vectorizer, vectors
