"""
Experiment to evaluate text summarization using different values of *k* top
sentences.
"""

import pandas as pd
import numpy as np
import spacy
from utils.paths import *
from utils import utils, plotter
from typing import Iterable
from transformers import BertModel, BertTokenizer


def evaluate_summarization_candidates(
    corpus_overviews: list,
    candidates: Iterable,
    embedder: BertModel = None,
    spacy_model: spacy.language.Language = None,
    tokenizer: BertTokenizer = None,
    **kwargs
):
    """Get an array of reduction rates and number of word pieces for a set of
    candidate number of sentences used to summarize each book overview (must
    be a list or array-like of integers).

    :param corpus_overviews: Book overviews.
    :type corpus_overviews: list or array-like
    :param candidates: List of number of candidates to evaluate.
    :type candidates: Iterable
    :param embedder: Instance of `SentenceTransformer`. If `None`, it loads the default
     pretrained sentence transformer model. Defaults to None.
    :type embedder: SentenceTransformer, optional
    :param spacy_model: Pretrained tokenizer model. If `None`, it
     loads the default model. Defaults to None.
    :type spacy_model: spacy.language.Language, optional
    :param tokenizer: Instance of `BertTokenizer` class. If `None`, it loads the
     pretrained tokenizer of 'bert-base-uncased'.
    :type tokenizer: BertTokenizer, optional
    :return: Summarized overviews with at most `candidates` sentences.
    :rtype: list
    """
    if not embedder:
        # Get the default sentence transformer.
        embedder = utils.get_sentence_transformer()
    if not spacy_model:
        # Get the default spacy model.
        spacy_model = spacy.load("en_core_web_sm")
    if not tokenizer:
        # Get the default BERT Tokenizer.
        tokenizer = utils.get_bert_tokenizer()

    # Array containing the word piece tokens for each candidate.
    wp_tokens = np.empty((0, len(candidates)), int)
    # Array containing the reduction rate of word pieces for each candidate.
    wp_reduction = np.empty((0, len(candidates)), float)
    for doc in corpus_overviews:
        # If the document is empty, skip it.
        if not doc:
            continue
        # We use a high value of k to retrieve all ranked sentences.
        sentences = utils.get_top_k_sentences(
            doc, k=100, embedder=embedder, spacy_model=spacy_model, **kwargs
        )

        # Number of word pieces of the original document.
        doc_original_tokens = len(utils.get_tokenized_text(doc, tokenizer))
        # Collection of word piece tokens for each summarized document using
        # the top k sentences.
        doc_wp_tokens = np.array(
            [
                len(utils.get_tokenized_text(" ".join(sentences[:k]), tokenizer))
                for k in candidates
            ]
        )
        # Collection of word piece tokens reduction for each summarized document
        # using the top k sentences.
        doc_wp_reduction = np.array(
            [
                (doc_original_tokens / doc_sum_tokens) - 1
                for doc_sum_tokens in doc_wp_tokens
            ]
        )
        # Append the results obtained for this document to the general respective
        # arrays.
        wp_tokens = np.vstack([wp_tokens, doc_wp_tokens])
        wp_reduction = np.vstack([wp_reduction, doc_wp_reduction])

    return wp_tokens, wp_reduction


if __name__ == "__main__":
    df = pd.read_csv(PATH_BOOKS_PROCESSED, sep=",")

    # We only consider non-null overviews.
    corpus = df[df["overview"].notna()].overview.tolist()

    # Our candidates will be {2,3,...9}
    list_candidates = list(range(2, 10))

    # We indicate to not preserve the order, otherwise we would obtain the
    # original document splitted into sentences and it would not make sense
    # to rank them.
    arr_tokens, arr_reduction = evaluate_summarization_candidates(
        corpus, candidates=list_candidates, preserve_order=False
    )

    # Compute the average number of word piece tokens per summarization strategy
    avg_tokens_per_summarization_strategy = np.sum(arr_tokens, axis=0) // len(corpus)

    # Compute the average reduction rate of word piece tokens per summarization strategy
    avg_reduction_per_summarization_strategy = np.sum(arr_reduction, axis=0) / len(
        corpus
    )

    # Plot results.
    fig = plotter.plot_scatter_with_secondary_y_axis(
        list_candidates,
        avg_tokens_per_summarization_strategy,
        avg_reduction_per_summarization_strategy,
        fig_title="Reduction rate and number of word pieces as the number of sentences used to summarize the book overviews increases.",
        x_title="Number of top central sentences for summarization",
        y_title="Average number of word pieces",
        y2_title="Average reduction rate of word pieces",
    )

    fig.show()
