# Image Search Engine

## Table of contents

- [Description](#Description)
- [Archticture diagram](#Archticture-diagram)
- [How does it work](#How-does-it-work)
  - [Offline phase (training)](<#Offline-phase-(training)>)
  - [Online phase](#Online-phase)
- [How to use it](#How-to-use-it)
- [Technologies](#Technologies)

## Description

A deep-learning-based image search engine that uses deep learning to extract images features and builds an inverted index to perform an image-based search

## Archticture diagram

![Archticture diagram](./ISE%20Archticture.png)

## How does it work

it's constructed of 2 phases: offline phase and online phase

### Offline phase (training)

In this phase, the index is built to be used later to perform the search and this is done by the following steps:

1- Feature extractor based in **Inception ResNet V2** CNN is used to extract deep features of all the images in the dataset and write them to the desk/Database as `.npy` files.

2- An indexer is used to build an **inverted index** by clustering the features on a number of clusters (passed to the build method) using the **K-means** Algorithm and uses the centers of the clusters as the terms of the index which maps a cluster centers to the ids of the features that belong to this cluster and then this index is saved to the desk/Database as `.hdf5` file.

### Online phase

In this phase, the engine takes queries from users and uses the index to find the most similar images to it. each of which is associated with its similarity to the queried image following the following steps:

1- Again the feature extractor is used but this time to extract the deep features of the query.

2- The features are then passed to the index which first measures the similarity between the query features and the centers of the clusters using the **L2 similarity** measure and ranks the clusters based on the similarity to the query.

3- The index picks the top similar clusters (the number of clusters to consider is passed to the search method) and again measures the similarity with all data points associated with each of the clusters, ranks them, and returns the top k results (k is a parameter to the search method).

## How to use it

1- set up your `static/images/features/index paths` in the `.env` file.

2- run the `offline_training.py` file to run the offline phase.

3- use the `search.py` module to perform the search by passing the image to the method `Search.search()`.

_note: there is a simple flask server associated with the engine that can be used to perform the search directly with a web-based interface after finishing the training phase_

## Technologies

TensorFlow
Keras
Scikit-learn
Flask
