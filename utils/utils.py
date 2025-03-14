"""
Set of miscellaneous utilities used across the implementation.
"""
import logging
import pandas as pd
import numpy as np
import pickle  # Object serialization
import spacy
import nltk
import re

from typing import Union
from utils.lexrank import degree_centrality_scores
from pathlib import Path
from torch import Tensor, topk
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder, util

def get_logger(
    name: str,
    log_level: Union[int, str] = logging.INFO,
    stream_handler=True,
    file_handler=True,
) -> logging.Logger:
    """Returns logger object for module with name `name`.

    :param name: name of the logger
    :type name: str
    :param log_level: log level, defaults to logging.INFO
    :type log_level: Union[int, str], optional
    :param stream_handler: include stream handler, defaults to True
    :type stream_handler: bool, optional
    :param file_handler: include file handler, defaults to True
    :type file_handler: bool, optional
    :return: Logger object
    :rtype: logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Disable propagation to avoid log duplicates.
    logger.propagate = False

    # Return logger for module `name`, if it exists
    if len(logger.handlers):
        return logger

    # Create handlers: one console hanlder and one file handler,
    # according to function params
    handlers = []

    if stream_handler:
        handlers.append(logging.StreamHandler())
    
    if file_handler:
        handlers.append(logging.FileHandler(f'logs_{name}.log'))
    
    # log format
    log_format = logging.Formatter(
        '%(asctime)s - [%(process)d] - %(name)s - [%(funcName)s] - %(levelname)s - %(message)s'
    )

    for handler in handlers:
        # Define the severity level for each handler.
        handler.setLevel(log_level)

        # Set formatter for the handler
        handler.setFormatter(log_format)

        # Add handler to the logger
        logger.addHandler(handler)

    return logger


def remove_filename_from_path(out_filename: str, path_standard_format=False) -> str:
    """Attempts to remove filename from the provided path.

    :param out_filename: Filepath.
    :type out_filename: str
    :param path_standard_format: Indicates whether the path follows the standard
     format (backslash separator) or the slash separator, defaults to False.
    :type path_standard_format: bool, optional
    :return: The directory excluding the filename.
    :rtype: str
    """
    if path_standard_format:
        out_filename = out_filename.replace("\\", "/")
    return (
        out_filename
        if "/" not in out_filename
        else out_filename.replace(out_filename.split("/")[-1], "")
    )


def makedir(path: str, remove_filename=False, recursive=True, exist_ok=True) -> None:
    """Creates directory from path if not exists.

    :param path: Path of the directory to be created.
    :type path: str
    :param remove_filename: If set to True, it attempts to remove the filename from
     the path, defaults to False
    :type remove_filename: bool, optional
    :param recursive: Creates directories recursively (i.e., create necessary
     subdirectories if necessary), defaults to True
    :type recursive: bool, optional
    :param exist_ok: is set to False, arises an error if `path` directory exists,
     defaults to True
    :type exist_ok: bool, optional
    """
    if "/" in path or "\\" in path:
        path = path if not remove_filename else remove_filename_from_path(path)
        Path(path).mkdir(parents=recursive, exist_ok=exist_ok)


def load_corpus(path_corpus: str, sep=",", **kwargs) -> pd.DataFrame:
    """Wrapper method of Pandas `read_csv` function to load book corpus.

    :param str path_corpus: Filepath to the corpus.
    :param sep: Delimiter to use, defaults to ','
    :type sep: str, optional
    :return: Corpus DataFrame
    :rtype: DataFrame
    """
    return pd.read_csv(path_corpus, sep=sep, **kwargs)


def get_dir_files_content(directory: str, file_extension=".txt") -> list:
    """Get the list of files in `directory` with extension `file_extension`.

    :param str directory: Directory in which the files are located. Recursive search
     is not allowed.
    :param file_extension: File extension, defaults to '.txt'
    :type file_extension: str, optional
    :return: List of tuples (filename, file content).
    :rtype: list
    """
    # This function retrieves the name of a file,
    # which matches the identifier of the book,
    # i.e., the`gr_book_id`.
    id_from_path = (
        lambda path: str(path).replace(directory, "").replace(file_extension, "")
    )
    # Return the list of pairs (book id, book overview) for
    # each review in the directory.
    return [
        [id_from_path(path), get_file_id_and_content(path)]
        for path in Path(directory + "\\").glob("*" + file_extension)
    ]


def get_file_id_and_content(filepath: str) -> str:
    """Returns all textual content in`filepath`

    :param filepath: Path of the text file to read.
    :return: Content of the file.
    :rtype: str
    """
    with open(filepath, "r", encoding="utf8") as file:
        data = file.read()
    return data


def append_overviews_to_data(
    df_overviews: pd.DataFrame,
    df_data: pd.DataFrame,
    overview_index: str = "gr_book_id",
    data_index: str = "gr_book_id",
    merge_option: str = "right",
) -> pd.DataFrame:
    """Merge `df_overviews` with `df_data` using column identifiers `overview_index` and
    `data_index`, respectively. We allow books with no overviews, hence *right join*
    is the most suitable operation.

    :param pd.DataFrame df_overviews: Book overviews.
    :param pd.DataFrame df_data: Book data (e.g., title, authors, etc.)
    :param overview_index: Column or index level names to join on in the left DataFrame,
     defaults to 'gr_book_id'.
    :type overview_index: str, optional
    :param data_index: Column or index level names to join on in the right DataFrame,
     defaults to 'gr_book_id'.
    :type data_index: str, optional
    :param merge_option: Type of merge operation, to be performed can be one of {'left',
     'right', 'outer', 'inner'}, to 'right'.
    :type merge_option: str, optional
    :return: A DataFrame of the two merged objects.
    :rtype: pd.DataFrame
    """
    return pd.merge(
        df_overviews,
        df_data,
        left_on=overview_index,
        right_on=data_index,
        how=merge_option,
    ).drop(columns=[overview_index])


def generate_dataframe_from_sparse_txts(
    base_dir: str, path_standard_format: bool = False, out_filename: str = None
) -> pd.DataFrame:
    """Generates a dataframe from all *txt* files located in `base_dir`. The dataframe
    features two columns: `gr_book_id`, an identifier that is retrieved from the name of
    each *txt* file, and `overview`, containing all information included in the *txt* file.

    :param str base_dir: Directory in which the *txt* files for the book overviews
     are located.
    :param path_standard_format: Indicates whether the path follows the standard
     format (backslash separator) or the slash separator, defaults to False.
    :type path_standard_format: bool, optional
    :param out_filename: Path for the output file, defaults to None.
    :type out_filename: str, optional
    :return: Returns a dataframe from all txt files located in `base_dir`.
    :rtype: pd.DataFrame
    """
    if not path_standard_format:
        base_dir = base_dir.replace("/", "\\")
    df = pd.DataFrame(
        data=get_dir_files_content(base_dir), columns=["gr_book_id", "overview"]
    )
    df.gr_book_id = df.gr_book_id.astype("int64")
    if out_filename:
        df.to_csv(f"{base_dir}{out_filename}.csv", sep=",")
    return df


def pickle_load(filename: str):
    """Read and return an object from the pickle data stored in a file.

    :param str filename: Filename of the file to be loaded.
    :return: deserialized data.
    :rtype: Any
    """
    with open(filename, "rb") as fIn:
        data = pickle.load(fIn)
    return data


def pickle_dump(
    data, filename: str, protocol: int = pickle.HIGHEST_PROTOCOL, **kwargs
) -> None:
    """_summary_

    :param data: Data to serialize.
    :type data: Any
    :param filename: output filename.
    :type filename: str
    :param protocol: pickle serialization protocol, defaults to pickle.HIGHEST_PROTOCOL
    :type protocol: int, optional
    """
    # Create directory if it does not exist.
    makedir(filename, remove_filename=True)
    with open(filename, "wb") as fOut:
        pickle.dump(data, fOut, protocol=protocol, **kwargs)


def store_embeddings(
    corpus_embeddings,
    out_filename="embeddings.pkl",
    protocol=pickle.HIGHEST_PROTOCOL,
    **kwargs,
):
    """Utility to dump embeddings (and other optional values indicated in the
    keyword arguments) to disk using *pickle*.

    :param corpus_embeddings: Tensor type data structure containing the embeddings
     for the corpus.
    :param out_filename: Path for the output file, defaults to 'embeddings.pkl'.
    :type out_filename: str, optional
    :param protocol: Protocol used for *pickle*, defaults to `pickle.HIGHEST_PROTOCOL`.
    """
    pickle_dump(
        {"embeddings": corpus_embeddings, **kwargs}, out_filename, protocol=protocol
    )


def load_embeddings(filename: str, return_dict_values=True):
    """Utility to load embeddings (and other optional stored values) from disk
    using *pickle*.

    :param str filename: Filename of the file to be loaded.
    :param return_dict_values: If set to True, returns the values just the values
     of the dictionary containing all stored data, defaults to True.
    :type return_dict_values: bool, optional
    :return: Loaded data
    """
    # Load embeddings and other stored information from disk
    stored_data = pickle_load(filename)
    return stored_data.values() if return_dict_values else stored_data


def split_text_into_sentences_spacy(
    text: str, spacy_model="en_core_web_sm"
) -> list[str]:
    """Splits text into sentences using the Spacy library. SpaCy builds a syntactic
    tree for each sentence, a robust method that yields more statistical information
    about the text than NLTK. It performs substancially better than NLTK when using
    not polished text.

    :param str text: Text to be splitted into sentences.
    :param spacy_model: Name of the spacy pretrained model used to split text into
     sentences, defaults to 'en_core_web_sm'.
    :type spacy_model: str, optional
    :return: List of sentences in `text`.
    :rtype: list
    """
    sent_tok = spacy.load(spacy_model)
    return [i.text for i in sent_tok(text).sents]


def split_text_into_sentences_nltk(text: str) -> list[str]:
    """Splits text into sentences using the NLTK library.

    :param str text: Text to be splitted into sentences.
    :return: List of sentences in `text`.
    :rtype: list
    """
    return nltk.sent_tokenize(text)


def clean_book_title(
    title: str,
    remove_quotation_marks: bool = True,
    remove_saga_info: bool = False,
    remove_saga_number: bool = True,
) -> str:
    """Applies several transformations to a book title to remove noisy data that
    can potentially affect the performance of the embedding strategies.

    :param str title: Book title in plain text.
    :param remove_quotation_marks: If set to True, attempts to remove the quotation
     marks enclosing the book title, to True.
    :type remove_quotation_marks: bool, optional
    :param remove_saga_info: If set to True, attempts to remove information concerning
     the book saga, defaults to False.
    :type remove_saga_info: bool, optional
    :param remove_saga_number: If set to True, attempts to remove the saga number,
     defaults to True.
    :type remove_saga_number: bool, optional
    :return: Processed book title.
    :rtype: str
    """
    # If the book title is surrounded by quoting marks
    if remove_quotation_marks and re.match(r'(^(").+(")$)|((^(\').+(\')$))', title):
        # we delete them.
        title = title[1:-1]

    # Remove book saga information.
    if remove_saga_info:
        title = re.sub(r"\(.+ +\#.+\).*", "", title)

    # Remove the number of the book in the saga.
    if remove_saga_number and re.search(r"\(.+ +\#.+\).*", title):
        saga_info = re.search(r"\(.+ +\#.+\).*", title).group(0)
        saga_info_new = re.sub(r"[\, ] *\#.+", ")", saga_info)
        title = title.replace(saga_info, saga_info_new)
    return title


def fix_punctuation(overview: str) -> str:
    """Attempts to fix some of the identified punctuation issues present in the
    book overviews.

    It is a common issue to find overviews with the following punctuation flaw:
        * "[...] word.Word [...]"
        * "[...] word!Word [...]"
        * "[...] word?Word [...]"

    That is to say, spacing after periods, exclamation and question marks is not
    correctly applied. This lead to some issues when splitting the text into sentences,
    specially using the NLTK library. Furthermore, it may have other adverse effects
    on the embedding process (e.g., due to faulty tokenization).

    :param str overview: Book overview in plain text.
    :type overview: str
    :return: str
    :rtype: Book overview without the identified punctuation flaws.
    """
    punct_errs = re.finditer("[a-z'][\.!?][A-Z]", overview)
    for error in punct_errs:
        str_error = error.group()
        # Correct the error by adding a space between the
        str_fix = str_error.replace(str_error[1], f"{str_error[1]} ")
        overview = overview.replace(str_error, str_fix)
    return overview


def clean_overview(overview: str) -> str:
    """Applies several transformations to a book overview to remove noisy data
    that can potentially affect the performance of the embedding strategies.
    One must be careful when applying transformations to the whole corpus because
    the odds for negative side-effects are high. Here, we attempt to solve some of the
    problems spotted that, in our tests, should not have any noticeable negative effect
    on any book overview.

    :param str overview: Book overview in plain text.
    :type overview: str
    :return: Processed book overview.
    :rtype: str
    """
    # Dictionary containing a bunch of transformations
    # to be applied to the text.
    STR_TRANSFORMATIONS = {"--": " ", ". . .": " ", "…": "... "}
    # All characters to be removed from the book overview.
    chars_to_delete = '"#$()~{|}`[\]@<=>^*“”'
    # Maketrans translation table usable for str.translate()
    text_cleanser = str.maketrans("", "", chars_to_delete)
    # Characters in `chars_to_delete` are removed upon calling
    # the translate method on a string.
    overview = overview.translate(text_cleanser)
    # Strings matching any of the keys in the dictionary
    # will be replaced with the the corresponding string
    # indicated in the value mapped to that key.
    for k, v in STR_TRANSFORMATIONS.items():
        overview = overview.replace(k, v)
    # Remove extra blank spaces.
    overview = re.sub(" +", " ", overview)
    # Fix some puctuation issues.
    overview = fix_punctuation(overview)
    # Return the processed book overview.
    return overview


def prepare_input_encoder(
    encoding_strategy: str, corpus: pd.DataFrame, return_input_encoder=True
) -> Union[list[str], pd.DataFrame]:
    """Formats the input to the encoder using the features indicated in
    `encoding_strategy`. If `encoding_strategy` takes a wrong value this method
    is likely to fail. Current supported features are 'title', 'authors' and
    'overview'.
    :param str encoding_strategy: The encoding strategy must be a string containing
    the names of the features to include into the input of the encoder, each of them
    separated by an underscore ('_'). For example, if you were to use the title and
    the overview as the encoding strategy, `encoding_strategy` must be either
    `title_overview` or `overview_title`.
    :param str path_df: Path in which the dataframe is located.
    :param return_input_encoder: Return just the collection of inputs to the encoder,
    defaults to True
    :type return_input_encoder: bool, optional
    :return: If `return_input_encoder`, returns the collection of inputs to the encoder.
    Otherwise, it returns Dataframe including a new column `input_encoder` with the
    format indicated in `encoding_strategy`
    :rtype: Union[list[str], pd.DataFrame]
    """
    ORDER_INPUT_ENCODER = dict(title=1, authors=2, overview=3)
    # Read dataframe from disk and fill NaN values with an empty
    # string for easier data management.
    df = corpus.fillna("")
    # Compute the set of features to include in the encoder input
    options = set(encoding_strategy.split("_"))
    # Order the option values according to the values in ORDER_INPUT_ENCODER
    opt_idx = [tuple([option, ORDER_INPUT_ENCODER[option]]) for option in options]
    sorted_o = [
        sorted_option[0] for sorted_option in sorted(opt_idx, key=lambda x: x[1])
    ]
    # Overviews for a book are not always available, thus we have to control
    # that we are not adding an extra [SEP] token to the input of the encoder.
    if "overview" in encoding_strategy:
        # Copy of the sorted options list.
        sorted_o_c = sorted_o[:]
        # Remove overview from the sorted option list.
        # It is necessarily always going to be the one in
        # the tail of the list because it has assigned the
        # least priority.
        sorted_o_c.pop()
        # If there are more elements after popping overview
        # from the list of options
        if sorted_o_c:
            append_overview = lambda b: "[SEP]" + b if b else ""
            # Join the values in the dataframe by means of the
            # special [SEP] token, checking whether there is an
            # overview for the book beforehand to avoid appending
            # the unwanted [SEP] token.
            df["input_encoder"] = [
                "[SEP]".join([book[opt] for opt in sorted_o_c])
                + append_overview(book["overview"])
                for _, book in df.iterrows()
            ]
        else:
            # If there are not any elements after popping
            # overview, overview is the only option.
            df["input_encoder"] = df["overview"]
    else:
        # If overview is not included in the encoding strategy,
        # join the values in the dataframe by means of the special
        # [SEP] token
        df["input_encoder"] = [
            "[SEP]".join([book[opt] for opt in sorted_o]) for _, book in df.iterrows()
        ]
    return df["input_encoder"].tolist() if return_input_encoder else df


def topk_cos_sim(
    query_embedding: Tensor, embeddings: Tensor, top_k: int
) -> tuple[list, list]:
    """Get the indices and the cosine similarity score of the `top_k` most similar embeddings
    to `query_embedding` in `embeddings`.

    :param query_embedding: Query embedding.
    :type query_embedding: Tensor
    :param embeddings: Corpus embeddings.
    :type embeddings: Tensor
    :param int top_k: Top k most similar to retrieve according to cosine similarity
     score.
    :return: List of scores and list of indexes of the top k results.
    :rtype: (list, list)
    """
    # We use cosine-similarity and torch.topk to find the highest k scores
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    return topk(cos_scores, k=top_k)


def get_bert_model(transformer="bert-base-uncased") -> BertModel:
    """Get an instance of the class `BertModel` for transformer `trasformer`.

    :param transformer: model `id` of a checkpoint hosted inside a
     model repo on huggingface.co, defaults to "bert-base-uncased"
    :type transformer: str, optional
    :return: Instance of `BertModel` class
    :rtype: BertModel
    """
    return BertModel.from_pretrained(transformer)


def get_bert_tokenizer(transformer="bert-base-uncased") -> BertTokenizer:
    """Get an instance of the class `BertTokenizer` for transformer `trasformer`.

    :param transformer: Model `id` of a pretrained tokenizer hosted inside a
     model repo on huggingface.co, defaults to "bert-base-uncased"
    :type transformer: str, optional
    :return: Instance of `BertTokenizer` class
    :rtype: BertTokenizer
    """
    return BertTokenizer.from_pretrained(transformer)


def get_sentence_transformer(
    transformer="paraphrase-distilroberta-base-v2", max_seq_length=None, **kwargs
) -> SentenceTransformer:
    """Wrapper function to get a `SentenceTransformer`.

    :param transformer: The model `id` of a pretrained `SentenceTransformer` hosted
     inside a model repo on HuggingFace. Defaults to 'paraphrase-distilroberta-base-v2'.
    :type transformer: str, optional
    :param int max_seq_length: Property to get the maximal input sequence length
     for the model. Longer inputs will be truncated. Defaults to None.
    :return: a SentenceTransformer model that can be used to map sentences / text
     to embeddings.
    :rtype: SentenceTransformer
    """
    transformer = SentenceTransformer(transformer, **kwargs)
    if max_seq_length:
        transformer.max_seq_length = max_seq_length
        logging.info("New maximum transformer input length:", transformer.max_seq_length)
    return transformer


def get_crossencoder(
    crossencoder="cross-encoder/stsb-distilroberta-base", **kwargs
) -> CrossEncoder:
    """Wrapper function to get a `CrossEncoder`.

    :param crossencoder: Any model name from Huggingface Models repository that can
     be loaded with AutoModel. Defaults to 'cross-encoder/stsb-distilroberta-base'.

    :type crossencoder: str, optional
    :return: a CrossEncoder that takes exactly two sentences/texts as input and predicts
     a score for this sentence pair.It can for example predict the similarity of the
     sentence pair on a scale of 0 ... 1.
    :rtype: CrossEncoder
    """
    return CrossEncoder(crossencoder, **kwargs)


def get_tokenized_text(text: str, tokenizer=None) -> list[str]:
    """Get the list of WordPiece tokens in `text`

    :param text: Text to tokenize
    :type text: str
    :param tokenizer: Instance of `BertTokenizer` class. If `None`, it loads the
     pretrained tokenizer of 'bert-base-uncased'.
    :type tokenizer: BertTokenizer, optional
    :return: list of WordPiece tokens.
    :rtype: list
    """
    if not tokenizer:
        # Get the by default BertTokenizer.
        tokenizer = get_bert_tokenizer()
    ids = tokenizer.encode(text)
    # Return list of tokens in `text`.
    return tokenizer.convert_ids_to_tokens(ids)


def compute_avg_wordpiece_tokens(corpus: list, tokenizer: BertTokenizer = None) -> int:
    """Compute the average number of WordPiece tokens in a list of documents, `corpus`.

    :param corpus: List of textual documents.
    :type corpus: list or array-like
    :param tokenizer: Instance of `BertTokenizer` class. If `None`, it loads the
     pretrained tokenizer of 'bert-base-uncased'.
    :type tokenizer: BertTokenizer, optional
    :return: Average number of WordPiece tokens of the documents in `corpus`.
    :rtype: int
    """
    if not tokenizer:
        # Get the by default BertTokenizer.
        tokenizer = get_bert_tokenizer()
    len_corpus = 0
    for text in corpus:
        len_corpus += len(get_tokenized_text(text, tokenizer))
    # Return truncated number of average number
    # WordPiece tokens in `corpus`.
    return len_corpus // len(corpus)


def get_top_k_sentences(
    document: str,
    k=5,
    embedder: SentenceTransformer = None,
    spacy_model: spacy.language.Language = None,
    preserve_order=True,
) -> list[str]:
    """Get the top `k` most meaningful sentences of `document`.

    :param document: a document in plain text
    :type document: str
    :param k: Top k sentences to return, defaults to 5
    :type k: int, optional
    :param embedder: Instance of `SentenceTransformer`. If `None`, it loads the
     default pretrained sentence transformer model. Defaults to None.
    :type embedder: SentenceTransformer, optional
    :param spacy_model: Pretrained tokenizer model. If `None`, it
     loads the default model. Defaults to None.
    :type spacy_model: spacy.language.Language, optional
    :param preserve_order: Preserve the order of the top k sentences with respect
     to the original document to conserve spatial dependencies between sentences.
     Defaults to True.
    :type preserve_order: bool, optional
    :return: Top `k` most meaningful sentences of `document`.
    :rtype: list
    """
    if not embedder:
        # Get the default sentence transformer.
        embedder = get_sentence_transformer()
    if not spacy_model:
        # Get the default spacy model.
        spacy_model = spacy.load("en_core_web_sm")
    if not document:
        return []
    # Split the document into sentences
    sentences = [i.text for i in spacy_model(document).sents]
    # Compute the embeddings for all sentences
    embeddings = embedder.encode(sentences, convert_to_tensor=True)
    # Compute the pair-wise cosine similarities
    cos_scores = util.pytorch_cos_sim(embeddings, embeddings).numpy()
    # Compute the centrality for each sentence
    centrality_scores = degree_centrality_scores(cos_scores, threshold=None)
    # Argsort in descending order, so that the first element
    # corresponds to the index of the most relevant sentence.
    central_sent_ids = np.argsort(-centrality_scores)[:k]
    # Preserve the order of the  top k sentences with
    # respect to the original document
    if preserve_order:
        central_sent_ids.sort()
    # Return the top `k` most meaningful sentences of `document`.
    return [sentences[idx].strip() for idx in central_sent_ids]


def summarize_corpus_overviews(
    corpus_overviews: list,
    top_k: int = 5,
    embedder: SentenceTransformer = None,
    spacy_model: spacy.language.Language = None,
    **kwargs,
) -> list[str]:
    """Apply unsupervised Text Summarization techniques to obtain representations for
    the most meaningful sentences for each document in `corpus_overviews`.

    :param corpus_overviews: Book overviews.
    :type corpus_overviews: list or array-like
    :param top_k: Number of sentences that will have each overview. Defaults to 5.
    :type top_k: int, optional
    :param embedder: Instance of `SentenceTransformer`. If `None`, it loads the default
     pretrained sentence transformer model. Defaults to None.
    :type embedder: SentenceTransformer, optional
    :param spacy_model: Pretrained tokenizer model. If `None`, it
     loads the default model. Defaults to None.
    :type spacy_model: spacy.language.Language, optional
    :return: Summarized overviews with at most `top_k` sentences.
    :rtype: list
    """
    if not embedder:
        # Get the default sentence transformer.
        embedder = get_sentence_transformer()
    if not spacy_model:
        # Get the default spacy model.
        spacy_model = spacy.load("en_core_web_sm")
    # Return a collection of summarized overviews.
    # A summarized overview is the concatenation of
    # the `top_k` most central sentences.
    return [
        " ".join(
            get_top_k_sentences(
                doc, k=top_k, embedder=embedder, spacy_model=spacy_model, **kwargs
            )
        )
        for doc in corpus_overviews
    ]
