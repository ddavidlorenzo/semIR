"""
Set of plotting utilities used across the implementation.
"""
import logging
import numpy as np
import pandas as pd
import torch
from sentence_transformers.util import pytorch_cos_sim
from transformers import BertModel, BertTokenizer
from typing import Literal, Union
from utils import utils

# SummaryWriter to export embeddings to visualize them
# with TensorBoard.
from torch.utils.tensorboard import SummaryWriter

# Sklearn for dimesionality reduction.
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_scatter_with_secondary_y_axis(
    x, y, y2, fig_title="", x_title="", y_title="", y2_title=""
) -> go.Figure:
    """Plot scatter plot with secondary y axis."""
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter(x=x, y=y, name=y_title),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=x, y=y2, name=y2_title),
        secondary_y=True,
    )
    # Add figure title
    fig.update_layout(title_text=fig_title)
    # Set x-axis title
    fig.update_xaxes(title_text=x_title)
    # Set y-axes titles
    fig.update_yaxes(title_text=y_title, secondary_y=False)
    fig.update_yaxes(title_text=y2_title, secondary_y=True)

    return fig


def project(
    X: np.ndarray,
    n_components: int,
    reduction_strategy: Literal["pca", "tsne"] = "pca",
    random_state: int = None,
    _tsne_perplexity: float = 5,
    _tsne_learning_rate: float = 10,
    _tsne_n_iter: int = 3000,
    **kwargs,
) -> tuple[np.ndarray, Union[PCA, TSNE]]:
    """Apply dimensionality reduction on an array-like input X using reduction
    techniques like the Principal Component Analysis (PCA) and the T-distributed
    Stochastic Neighbour Embedding (t-SNE).  The default values for the perplexity,
    learning rate and number of iterations have been empirically tuned to
    those that produced acceptable results consistently.

    :param X: array of features.
    :type X: list or array-like
    :param n_components: Dimension of the new embedded space (e.g., 2 for
     2D visualization, 3 for 3D visualization).
    :type n_components: int
    :param random_state: Random state for dimensionality reduction techniques,
     defaults to None
    :type random_state: int, optional
    :param reduction_strategy: Reduction strategy to choose. Can be either
     'pca' or 'tsne'. Defaults to 'pca'
    :type reduction_strategy: Literal["pca", "tsne"], optional
    :param _tsne_perplexity: The perplexity is related to the number of
     nearest neighbors that is used in other manifold learning algorithms.
     Defaults to 5.
    :type _tsne_perplexity: int, optional
    :param _tsne_learning_rate: The learning rate for t-SNE is usually
     in the range [10.0, 1000.0]. Defaults to 10.
    :type _tsne_learning_rate: int, optional
    :param _tsne_n_iter: Maximum number of iterations without progress
     to abort optimization process, defaults to 3000.
    :type _tsne_n_iter: int, optional
    :raises ValueError: Raise `ValueError` if `reduction_stragy` takes
     an ilegal value.
    :return: projected data.
    :rtype: tuple(np.ndarray, Union[PCA, TSNE])
    """
    # Remove noisy characters from the reduction strategy option.
    reduction_strategy = reduction_strategy.replace(" ", "").replace("-", "").lower()
    if reduction_strategy == "pca":
        # Reduce embeddings dimensionality using PCA.
        dim_compressor = PCA(
            n_components=n_components, random_state=random_state, **kwargs
        )
    elif reduction_strategy == "tsne":
        # Reduce embeddings dimensionality using t-SNE.
        dim_compressor = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=_tsne_perplexity,
            learning_rate=_tsne_learning_rate,
            n_iter=_tsne_n_iter,
            **kwargs,
        )
    else:
        raise ValueError(
            f"'{reduction_strategy}' is not a valid reduction strategy."
            + " Current available options are 'pca' and 'tsne'"
        )
    data = dim_compressor.fit_transform(X)
    return data, dim_compressor


def assert_n_dimensions(n_dimensions: Literal[2, 3]) -> None:
    try:
        N_DIM_VALUES = frozenset([2, 3])
        assert n_dimensions in N_DIM_VALUES
    except AssertionError as ae:
        msg_error = (
            f'Invalid value for parameter "n_dimensions". '
            f"Got: {n_dimensions}. Expected: one of {N_DIM_VALUES}."
        )
        logging.error(msg_error)
        raise ValueError(msg_error)


def scatter(
    X: Union[list, np.ndarray],
    n_dimensions: Literal[2, 3] = 2,
    color_text: str = None,
    random_state: int = None,
    reduction_strategy: Literal["pca", "tsne"] = "pca",
    marker_size: int = 12,
    _graph_showlegend: bool = True,
    tsne_params: dict = {},
    **scatter_params,
) -> tuple[go.Figure, Union[PCA, TSNE]]:
    """Visualize BERT embeddings in a 2D scatterplot using dimensionality
    reduction techniques like Principal Component Analysis (PCA) and the T-distributed
    Stochastic Neighbour Embedding (t-SNE).

    :param X: array of features.
    :type X: list or array-like
    :param color_text: Label for plotly scatterplot `color` attribute.
     Defaults to None
    :type color_text: str, optional
    :param random_state: Random state for dimensionality reduction techniques,
     defaults to None
    :type random_state: int, optional
    :param reduction_strategy: Reduction strategy to choose. Can be either
     'pca' or 'tsne'. Defaults to 'pca'
    :type reduction_strategy: str, optional
    :param _graph_showlegend: Show legend, defaults to True
    :type _graph_showlegend: bool, optional
    :param tsne_params: Additional parameters for t-SNE,
     defaults to {}
    :type scatter_params: dict, optional
    """
    assert_n_dimensions(n_dimensions)

    # Explicit cast to np.ndarray.
    X = np.array(X)
    dim_compressor = None
    if X.shape[1] > n_dimensions:
        # Get data in the new embedded space.
        X, dim_compressor = project(
            X,
            n_dimensions,
            reduction_strategy=reduction_strategy,
            random_state=random_state,
            **tsne_params,
        )

    # Labels for the graph axis.
    fig_labels = {str(dim): f"Dim {dim + 1}" for dim in range(n_dimensions)}

    # If there is a label for the `color` property
    if color_text:
        # we add it to the labels.
        fig_labels["color"] = color_text

    if n_dimensions == 2:
        # Create a 2D scatter plot with the projected data.
        # x=0, y=1 will take the values for the first and
        # second dimension of the transformed data.
        fig = px.scatter(X, x=0, y=1, opacity=0.6, labels=fig_labels, **scatter_params)
    else:
        # Create a 3D scatter plot with the projected data.
        # x=0, y=1, z=2 will take the values for the first,
        # second and third dimension of the transformed data.
        fig = px.scatter_3d(
            X, x=0, y=1, z=2, opacity=0.6, labels=fig_labels, **scatter_params
        )

    # Change the position and the size of the markers.
    fig.update_traces(textposition="top center", marker=dict(size=marker_size))

    # Update legend title text, and whether to show legend or not.
    fig.update_layout(showlegend=_graph_showlegend, legend_title_text="Colour legend")
    # Show figure.
    return fig, dim_compressor


def histogram_embeddings_nn(data: list):
    """Plot histogram for the nearest neighbors of an embedding.

    :param data: Two dimensional array containing a collection of similar
     words (first component), list of similarity scores (second component),
     and a list of labels (third component).
    :type data: list or array-like
    """
    # Get labels from data.
    labels = [d[2] for d in data]

    # Create an histogram with data. "x" are the list of similar words
    # "y" are the list of scores.
    fig = px.histogram(
        data,
        x=0,
        y=1,
        color=labels,
        title=f"Cosine similarity scores for the most similar words",
        labels={"0": "Similar word", "1": "cosine similarity"},
    )

    # Update name of the y axes.
    fig.update_yaxes(title="Cosine similarity")

    # Update legend title text.
    fig.update_layout(legend_title_text="Query word")

    # Show figure.
    return fig


def heatmap(data, color_continuous_scale: str = None, **kwargs) -> go.Figure:
    """Plot heatmap. Wrapper of the plotly `imshow` function.

    :param data: 2D data to be plotted.
    :type data: array-like
    :param color_continuous_scale: Colour scale to be used in the heatmap,
     defaults to None
    :type color_continuous_scale: str, optional
    """
    fig = px.imshow(data, color_continuous_scale=color_continuous_scale, **kwargs)
    return fig


def plot_bert_embs_nn(
    model: BertModel,
    vocab: dict,
    input_words: list,
    k: int = 5,
    n_dimensions: Literal[2, 3] = 2,
    reduction_strategy: Literal["pca", "tsne"] = "pca",
    random_state: int = None,
) -> tuple[go.Figure, go.Figure, Union[PCA, TSNE]]:
    """Visualize the `k` most similar words in `vocab` in the 2D or 3D
    embedding space.

    :param model: Bert pretrained model.
    :type model: BertModel
    :param vocab: Tokenizer vocabulary.
    :type vocab: dict
    :param input_words: Words, the KNN of which are to be calculated and
     displayed.
    :type input_words: Iterable
    :param k: number of similar words to visualize. By default, 5
    :type k: int, optional
    :param n_dimensions: Visualize BERT embeddings either in '2d' or
     '3d', defaults to '3d'
    :type n_dimensions: str, optional
    :param reduction_strategy: Reduction strategy to choose. Can be either
     'pca' or 'tsne'. Defaults to 'pca'
    :type reduction_strategy: str, optional
    :param random_state: Random state for dimensionality reduction techniques,
     defaults to None
    :type random_state: int, optional
    :raises ValueError: Raise `ValueError` if `n_dimensions` takes an
     ilegal value.
    """
    assert_n_dimensions(n_dimensions)

    # Reverse dict for easier access to tokens.
    indices = {v: k for k, v in vocab.items()}
    model.eval()
    # Get the pretrained embeddings.
    embeddings = model.embeddings.word_embeddings.weight
    # Results matrix.
    result = np.empty((0, 4), dtype="object")
    # For each word in `input_words`
    for i, word in enumerate(input_words):
        word_idx = vocab[word]
        # Prepare the results for the query word: the word itself,
        # `None` value for the similarity score and the embedding of
        # the input word.
        first_row = np.array(
            [[word, None, embeddings[word_idx].detach().numpy(), "input words"]],
            dtype="object",
        )
        # Append the word to the plotter data.
        result = np.append(result, first_row, axis=0)
        # Compute the k most similar words to the input word.
        scores, ids = utils.topk_cos_sim(embeddings[word_idx], embeddings, k + 1)

        # Append, for each similar word, its index in the vocabulary,
        # the cosine-similarity score betweenc the query word and the
        # similar word, and the embedding for thecsimilar word.
        # The query word is eventually added to later as a label
        # to be able to group all elements by the query word.
        rows = np.array(
            [
                tuple(
                    [
                        indices[int(idx)],
                        float(score),
                        embeddings[int(idx)].detach().numpy(),
                        word,
                    ]
                )
                for score, idx in zip(scores, ids)
                if word_idx != idx
            ][:k],
            dtype="object",
        )

        # Append the NN embeddings of the input word to the results
        # matrix.
        result = np.append(result, rows, axis=0)

    # The collection of similar words are in the first column.
    similar_words = result[:, 0]
    # The collection of embeddings of the most similar words are
    # in the third column.
    similar_words_embeddings = result[:, 2].tolist()
    # The collection of labels are located in the fourth column.
    similar_words_labels = result[:, 3]

    # Prepare data for the histogram (query words must be prunned).
    histogram_data = [
        tuple([word, score, label])
        for word, score, label in zip(similar_words, result[:, 1], similar_words_labels)
        if word not in input_words
    ]

    # Scatterplot additional features (title, group of colours, and the
    # text to be displayed for each word).
    scatter_params = dict(
        title=f"{reduction_strategy.upper()} {n_dimensions}D Visualization for BERT {k}-NN embeddings",
        color=similar_words_labels,
        text=similar_words,
    )
    # Invoke the proper plotter according to `n_dimensions`.
    fig_scatter, dim_compressor = scatter(
        similar_words_embeddings,
        n_dimensions=n_dimensions,
        color_text="neighbour to",
        random_state=random_state,
        reduction_strategy=reduction_strategy,
        **scatter_params,
    )

    # Plot histogram of the queries' most similar words.
    fig_hist = histogram_embeddings_nn(histogram_data)
    return fig_hist, fig_scatter, dim_compressor


def heatmap_embeddings(
    model: BertModel,
    tokenizer: BertTokenizer,
    data: pd.DataFrame,
    polysemous_word: str,
    color_continuous_scale: str = None,
) -> tuple[go.Figure, np.ndarray, list[str]]:
    """Plot heatmap of the cosine similarity of all different contextual embeddings
    of `polysemous_word` in `data`.

    :param model: Bert pretrained model.
    :type model: BertModel
    :param tokenizer: Bert precomputed tokenizer.
    :type tokenizer: BertTokenizer
    :param data: Test data for WSD evaluation.
    :type data: pd.DataFrame
    :param polysemous_word: Polysemous word.
    :type polysemous_word: str
    :param color_continuous_scale: Colour scale to be used in the heatmap,
     defaults to None
    :type color_continuous_scale: str, optional
    :return: The collection of the different contextual embeddings, along
     with the labels.
    """
    # Filter dataset by `polisemous_word`
    data = data[data["polysemy_word"] == polysemous_word]
    context_embeddings = []
    # Labels
    labels = []
    sentences = []
    with torch.no_grad():
        # For each sentence containing `polysemous_word` as the
        # target word
        for record in data.to_dict("records"):
            current_sentence = record["sentence/context"]
            sentences.append(current_sentence)
            # Tokenize sentence.
            ids = tokenizer.encode(current_sentence)
            tokens = tokenizer.convert_ids_to_tokens(ids)
            # Obtain the learned weights of BERT's model.
            bert_output = model.forward(
                torch.tensor(ids).unsqueeze(0), encoder_hidden_states=True
            )
            # Get the weights of the last layer.
            final_layer_embeddings = bert_output[0][-1]

            # Look for the polysemous word in the context sentence.
            for i, token in enumerate(tokens):
                if record["polysemy_word"].lower().startswith(token.lower()):
                    # Once it has been found, add the learned embedding
                    context_embeddings.append(final_layer_embeddings[i])
                    # And the label, which is the polysemous word followed
                    # by the serial number of the record.
                    labels.append(f'{token}_{record["sn"]}')

    # List to torch with the appropiate shape.
    context_embeddings = torch.stack(context_embeddings)

    # Compute matrix of similarity scores between the different contextual
    # representations of `polysemous_word`.
    cos_scores = pytorch_cos_sim(context_embeddings, context_embeddings)

    # Plot a heatmap with the results and the desired setup.
    fig = heatmap(
        cos_scores.detach().numpy(),
        title=f"Correlation matrix for different contextual embeddings of '{polysemous_word}'",
        labels=dict(color="Cosine similarity"),
        x=labels,
        y=labels,
        color_continuous_scale=color_continuous_scale,
    )

    # Print the contextual sentence corresponding to each label.
    logging.info(f"Context sentence for each contextual embedding of '{polysemous_word}'.\n")
    for label, sentence in zip(labels, sentences):
        logging.info(f"{label}: {sentence}")

    # Return the collection of the different contextual embeddings, along
    # with the corresponding labels.
    return fig, context_embeddings, labels


def write_embeddings_to_disk(
    model: BertModel,
    vocab: dict,
    out_dir: str = "runs/bert_embeddings",
    write_word_embeddings: bool = True,
    write_position_embeddings: bool = False,
    write_type_embeddings: bool = False,
) -> None:
    """Utility to write embeddings weights to disk that can be loaded with
      TensorBoard to visualize the embeddings.

    :param model: Bert pretrained model.
    :type model: BertModel
    :param vocab: Tokenizer vocabulary.
    :type vocab: dict
    :param out_dir: Directory to write the embeddings, defaults to
     'runs/bert_embeddings'.
    :type out_dir: str, optional
    :param write_word_embeddings: Write word embeddings to disk, defaults
     to True.
    :type write_word_embeddings: bool, optional
    :param write_position_embeddings: Write position embeddings to disk,
     defaults to False.
    :type write_position_embeddings: bool, optional
    :param write_type_embeddings: Write token type embeddings to disk,
     defaults to False.
    :type write_type_embeddings: bool, optional
    """
    # Writer will output to ./out_dir directory
    writer = SummaryWriter(out_dir)
    if write_word_embeddings:
        word_embedding = model.embeddings.word_embeddings.weight
        writer.add_embedding(
            word_embedding, metadata=vocab.keys(), tag=f"word embedding"
        )
    if write_position_embeddings:
        position_embedding = model.embeddings.position_embeddings.weight
        writer.add_embedding(
            position_embedding,
            metadata=np.arange(position_embedding.shape[0]),
            tag=f"position embedding",
        )

    if write_type_embeddings:
        token_type_embedding = model.embeddings.token_type_embeddings.weight
        writer.add_embedding(
            token_type_embedding,
            metadata=np.arange(token_type_embedding.shape[0]),
            tag=f"tokentype embeddings",
        )
    writer.close()
