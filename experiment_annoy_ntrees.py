"""
Experiment to test peedup-recall tradeoff depending on the number of trees
used in ANNOY.
"""

from semantic_search import SemanticSearch
from utils.paths import *
from utils import utils, plotter
from plotly.graph_objs import Figure

logger = utils.get_logger(__name__)

def evaluate_n_trees(
    queries: list, search_alg: SemanticSearch, k=5, verbose=True
) -> Figure:
    """Utility used to evaluate the speedup-recall tradeoff of ANNOY as
    the number of trees increases.

    :param queries: Collection of textual queries.
    :type queries: list or array-like
    :param search_alg: Semantic search object.
    :type search_alg: SemanticSearch
    :param k: Top k elements to consider in the comparison, defaults to 5
    :type k: int, optional
    :param verbose: Verbosity mode, defaults to True
    :type verbose: bool, optional
    """
    # Candidates to evaluate
    n_trees_list = list(range(128, 1028 + 64, 64))
    recall_list = []
    speedup_list = []
    for n_trees in n_trees_list:
        if verbose:
            logger.info("Evaluating ANNOY for 'n_trees': ", n_trees)
        # Setup ANNOY index with `n_trees` trees.
        search_alg.get_annoy_index(n_trees=n_trees)
        # Get recall and speedup for this configuration.
        recall, speedup = search_alg.test_annoy_performance(queries, k=k)
        recall_list.append(recall)
        speedup_list.append(speedup)
    # Print results
    fig = plotter.plot_scatter_with_secondary_y_axis(
        n_trees_list,
        speedup_list,
        recall_list,
        fig_title="Speedup-Recall trade-off as the number of trees increases",
        x_title="Nº trees",
        y_title="execution time",
        y2_title="recall",
    )

    return fig


if __name__ == "__main__":
    # Pretrained sentence transformer.
    pretrained_model = "paraphrase-distilroberta-base-v2"

    # Encoding strategy
    encoding_strategy = "title_overview"

    # Summarization strategy. We recommend using the top 5 sentences.
    summarization = "top5sent"  #'top4sent' #''

    # Filepath to store the computed embeddings on disk or path to disk in which
    # the embeddings are located. You may write the filepath you wish. If the directory
    # does not exist, we will attempt to create it.
    path_embs_cache = (
        f"{DIR_EMBEDDINGS}{pretrained_model}/{summarization}_{encoding_strategy}.pkl"
    )

    # Load the corpus from disk. Beware that the loaded corpus must be
    # consistent with the summarization technique you wish to use (e.g.,
    # for the 'top5sent' strategy, the dataset that must be is
    # 'books_processed_top5sent.csv')
    corpus = utils.load_corpus(PATH_BOOKS_TOP5S)

    # Number of top k nearest neighbours to retrieve by TF-IDF search.
    k = 5

    book_search = SemanticSearch(
        corpus,
        path_embs_cache=path_embs_cache,
        encoding_strategy=encoding_strategy,
    )

    # Guarantee experiment reproducibility.
    random_state = 0

    # Select the first 500 titles as queries.
    queries = corpus.sample(500, random_state=random_state).title.tolist()

    # Run experiment.
    fig = evaluate_n_trees(queries, book_search, k=k)

    fig.show()
