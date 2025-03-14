"""
This module defines the common interface for dense and lexical retrieval techniques.
"""
import pandas as pd
import os

from logging import Logger
from sentence_transformers import SentenceTransformer, CrossEncoder
from torch import Tensor
from utils.utils import (
    get_logger,
    get_sentence_transformer,
    get_crossencoder,
    load_embeddings,
    store_embeddings,
    prepare_input_encoder,
)


class Search:
    """This class serves as a common interface for dense and lexical retrieval
    techniques.

    Notes
    ------
    This class is not meant to be instantiated - use instead the specialized subclasses defined for
    dense and lexical IR accordingly.
    """

    @property
    def logger(self) -> Logger:
        """Logger object."""
        try:
            return self._logger
        except AttributeError:
            self.logger = get_logger(__name__)
        return self._logger

    @logger.setter
    def logger(self, logger: Logger) -> None:
        self._logger = logger

    @property
    def corpus(self) -> pd.DataFrame:
        """Book corpus. It should, at least, have the following columns
        (with the very same names):
        * `gr_book_id`: book unique identifier.
        * `title`: book titles.
        * `authors`: book authors.
        * `overview`: book overview.
        """
        return self._corpus

    @corpus.setter
    def corpus(self, corpus: pd.DataFrame) -> None:
        self._corpus = corpus

    @property
    def encoding_strategy(self) -> str:
        """Encoding strategy to use. The encoding strategy
        must be a string containing the names of the features to include into
        the input of the encoder, each of them separated by an underscore ('_').
        """
        return self._encoding_strategy

    @encoding_strategy.setter
    def encoding_strategy(self, encoding_strategy: str) -> None:
        self._encoding_strategy = encoding_strategy

    @property
    def path_embs_cache(self) -> str:
        """Filepath to store the computed embeddings or load the
        embeddings from.
        """
        return self._path_embs_cache

    @path_embs_cache.setter
    def path_embs_cache(self, path_embs_cache: str) -> None:
        self._path_embs_cache = path_embs_cache

    @property
    def path_biencoder(self) -> str:
        """Model id of a pretrained sentence transformer hosted on HuggingFace.
        """
        return self._path_biencoder

    @path_biencoder.setter
    def path_biencoder(self, path_biencoder: str) -> None:
        self._path_biencoder = path_biencoder

    @property
    def path_crossencoder(self) -> str:
        """Model `id` of a pretrained `CrossEncoder` hosted on 
         HuggingFace."""
        return self._path_crossencoder

    @path_crossencoder.setter
    def path_crossencoder(self, path_crossencoder: str) -> None:
        self._path_crossencoder = path_crossencoder

    @property
    def max_seq_length(self) -> int:
        """Property to get the maximal input sequence
         length for the model. Longer inputs will be truncated."""
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, max_seq_length: int) -> None:
        self._max_seq_length = max_seq_length

    @property
    def crossencoder(self) -> CrossEncoder:
        """Getter method for `crossencoder`.

        :return: Current pretrained Cross-Encoder. If None, the default
         Cross-Encoder is used.
        :rtype: CrossEncoder
        """
        try:
            return self._crossencoder
        except AttributeError:
            self.logger.info(
                f"Loading Cross-Encoder {self.path_crossencoder}"
            )
            self.crossencoder = get_crossencoder(self.path_crossencoder)
        return self._crossencoder

    @crossencoder.setter
    def crossencoder(self, crossencoder: CrossEncoder) -> None:
        """Set or update the Cross-Encoder to be used to re-rank the results
        retrieved by TF-IDF/Bi-encoder search. We recomend using either
        'cross-encoder/stsb-distilroberta-base' or any pretrained Cross-Encoder
        trained on  MS MARCO dataset.
        """
        self._crossencoder = crossencoder

    @property
    def biencoder(self) -> SentenceTransformer:
        """Getter method for `biencoder`.

        :return: Current pretrained Bi-Encoder. If None, the default Bi-Encoder
         is used.
        :rtype: SentenceTransformer
        """
        try:
            return self._biencoder
        except AttributeError:
            self.logger.info(
                f"Loading Bi-Encoder {self.path_biencoder}"
            )
            self.biencoder = get_sentence_transformer(
                self.path_biencoder, self.max_seq_length
            )

        return self._biencoder

    @biencoder.setter
    def biencoder(self, biencoder: SentenceTransformer) -> None:
        """Set or update the Bi-Encoder property"""
        self._biencoder = biencoder

    @property
    def embeddings(self) -> Tensor:
        """Get embeddings."""
        try:
            return self._embeddings
        except AttributeError:
            self.logger.info(
                f"Retrieving embeddings from disk for Bi-Encoder {self.path_biencoder} "
                f" with encoding strategy '{self.encoding_strategy}'"
            )
            self.embeddings = self._get_embeddings(self.path_embs_cache)
        return self._embeddings

    @embeddings.setter
    def embeddings(self, embeddings: Tensor):
        """Setter method for the corpus embeddings.

        :param embeddings: Corpus embeddings.
        :type embeddings: list or array-like.
        """
        self._embeddings = embeddings

    @property
    def input_encoder(self) -> list[str]:
        """Get input for the encoder.
        """
        try:
            return self._input_encoder
        except AttributeError:
            self.input_encoder = prepare_input_encoder(
                self.encoding_strategy, self.corpus
            )
        return self._input_encoder

    @input_encoder.setter
    def input_encoder(self, input_encoder: list[str]):
        """Setter method for input_encoder.

        :param input_encoder: Input data to the encoder.
        :type input_encoder: list or array-like.
        """
        self._input_encoder = input_encoder

    def __init__(
        self,
        corpus: pd.DataFrame,
        encoding_strategy="title_overview",
        path_biencoder: str = "paraphrase-distilroberta-base-v2",
        path_embs_cache: str = None,
        max_seq_length: int = 512,
        path_crossencoder: str = "cross-encoder/stsb-distilroberta-base",
    ):
        """Constructor for a `Search` instance.

        **Important**: this class is not meant to be directly instantiated.
        Use instead the specialized subclasses defined for dense and lexical
        IR accordingly.

        :param corpus: Book corpus. It should, at least, have the following columns
         (with the very same names):
         * `gr_book_id`: book unique identifier.
         * `title`: book titles.
         * `authors`: book authors.
         * `overview`: book overview.
        :type corpus: DataFrame
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
        :param path_embs_cache: Filepath to store the computed embeddings
         or load the embeddings from. Defaults to None.
        :type path_embs_cache: str, optional
        :param max_seq_length: Property to get the maximal input sequence
         length for the model. Longer inputs will be truncated. Defaults to 512.
        :type max_seq_length: int, optional
        :param path_crossencoder: Model `id` of a pretrained `CrossEncoder` hosted in 
         HuggingFace. Defaults to None.
        :type path_crossencoder: str, optional
        """
        self.corpus = corpus
        self.path_biencoder = path_biencoder
        self.path_crossencoder = path_crossencoder
        self.max_seq_length = max_seq_length
        self.path_embs_cache = path_embs_cache
        self.encoding_strategy = encoding_strategy

    def _get_embeddings(self, path_embs_cache=None) -> Tensor:
        """Get precomputed embeddings from `path_embs_cache` or compute
        semantic representations for each book in the corpus, that can be stored
        in `path_embs_cache`.

        :param path_embs_cache: Filepath to store the computed embeddings
         or load the embeddings from.
        :type path_embs_cache: str
        :return: Corpus embeddings
        :rtype: Tensor
        """
        # Check whether the embedding cache path exists
        if path_embs_cache and os.path.exists(path_embs_cache):
            embeddings, input_encoder = load_embeddings(path_embs_cache)
            self.logger.info(
                f"Fetched pre-computed embeddings from {path_embs_cache}"
            )
            self.input_encoder = input_encoder

        else:
            self.logger.info(
                "Unable to retrieve pre-computed embeddings from disk. "
                "Attempting to compute the embeddings with "
                f"the encoding strategy '{self.encoding_strategy}'. "
                "This may take a while."
            )
            embeddings = self.biencoder.encode(
                self.input_encoder, convert_to_tensor=True, show_progress_bar=True
            )
            # If there is a path to save the embeddings
            if path_embs_cache:
                # attempt to store them in disk.
                store_embeddings(
                    embeddings, corpus=self.corpus, out_filename=path_embs_cache
                )
        # Return corpus embeddings.
        return embeddings

    def _str_query_results(
        self, query: str, scores: list, indices: list, k: int
    ) -> str:
        """Retrieve query results in string format.

        :param str query: Query in plain text.
        :param scores: Collection of similarity scores
        :type scores: list or array-like
        :param indices: Collection of indices corresponding to the retrieved results.
        :type indices: list or array-like
        :param k: Number of most relevant documents retrieved.
        :type k: int
        :return: Query results in string format.
        :rtype: str
        """
        out = (
            "======================\n\n"
            f"Query: {query}"
            f"\nTop {k} most similar books in corpus:"
        )
        for score, idx in zip(scores, indices):
            doc = self.corpus.iloc[int(idx)]
            out += (
                f"\nTitle: {doc.title} -- (Score: {score:.4f}) (Goodreads Id: {doc.gr_book_id})"
                f"\nAuthors: {doc.authors}\nOverview: {doc.overview}\n\n"
            )
        return out
