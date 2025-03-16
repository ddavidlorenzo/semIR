.. Implementation of a real-time semantic retrieval system. documentation master file, created by
   sphinx-quickstart on Wed Jun  9 10:43:12 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for the implementation of a real-time semantic retrieval system.
======================================================================================
The increasingly overwhelming amount of available natural language motivates the pressing need to find efficient and reliable computational techniques capable of processing and analysing this type of data for the purpose of achieving human-like natural language understanding for a wide range of downstream tasks.

Over the last decade, Natural Language Processing (NLP) has seen impressively fast growth, primarily favoured by the increase in computational power and the progress on unsupervised learning in linguistics. Moreover, historically successful statistical language modeling techniques have been largely replaced by novel neural language modeling based on Deep Learning models, exhibiting an unprecedent level of natural language understanding, contributing to reduce the gap between human communication and computer understanding.

NLP is the key to solve many technological challenges. Among the large number of applications this field has, and since this dissertation has been entirely focused on the study of different strategies to encode salient information about natural language, the experimental part of this work is primarily devoted to the implementation of a semantic information retrieval system, delivering search latencies suitable for real-time similarity matching. 

Of course, the examined unsupervised pretrained representations have been trained on humongous data sets, devoting to that end massive amounts of computational resources, something we do not have the access to.  It is, however, not only a matter of computational power.  Gathering, preparing and processing data for a model to be fine-tuned is no easy task and requires knowledge, to a great extent, of both the underlying theoretical motivations and the specific implementation. Consequently, our work is pretty much aligned with the mentality in which these strategies sit on: to directly use the pretrained models for downstream tasks.

This document contains the documentation for all experiments conducted throughout the dissertation, including a thoroughly description of the implementation of the information retrieval system and the notebooks devoted to the dataset analysis and preprocessing and the analysis and visualization of BERT contextual embeddings.



.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   code
   search_usage.ipynb
   Visualization of BERT embeddings.ipynb
   Data preprocessing.ipynb
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
