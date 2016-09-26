# Learning Word Subsumption Projections

This is the implementation of the projection learning approach for learning word subsumptions, i.e., hyponyms and hypernyms, originally proposed in ([Fu et al., 2014]). The approach requires pre-trained word embeddings in the word2vec format and the list of subsumption examples to learn the projection matrix. This implementation uses [TensorFlow](https://www.tensorflow.org/).

[Fu et al., 2014]: http://aclanthology.info/papers/learning-semantic-hierarchies-via-word-embeddings

## Prerequisites

This implementation is designed for processing the Russian language, but there should be no problem in running it on any other language provided with the relevant datasets. However, for processing the Russian language, the following datasets are required:

* the trained word2vec model: [all.norm-sz100-w10-cb0-it1-min100.w2v],
* the set of semantic relations: [projlearn-ruwikt.tar.gz].

[projlearn-ruwikt.tar.gz]: http://ustalov.imm.uran.ru/pub/projlearn-ruwikt.tar.gz
[all.norm-sz100-w10-cb0-it1-min100.w2v]: https://s3-eu-west-1.amazonaws.com/dsl-research/wiki/w2v_export/all.norm-sz100-w10-cb0-it1-min100.w2v

## Models

The original approach learns a matrix such that transforms an input hyponym embedding vector into its hypernym embedding vector. A few variations of this approach have also been implemented. The following models are available:

* `baseline`, the original approach,
* `negative_hyponym` that penalizes the matrix to projecting the hypernyms back to the hyponyms,
* `negative_synonym` that penalizes the matrix to projecting the hypernyms back to the synonyms of the hypernyms,
* `positive_hypernym` that promotes the matrix to projecting the hyponym synonyms to the hypernyms.

## Training

Before any processing, certain things need to be precomputed, such as training and test sets, etc. For that, the `./prepare.py` script should be executed. On large embeddings, it might take a long time, but it is run only once.

Having the preparation script finished, the vector space should be separated into a number of clusters using the `./cluster.py` script. This is found to be very useful to improving the results, so it is not possible to continue without clustering. Usually, the clustering program automatically estimates the number of clusters using the [silhouette method](https://en.wikipedia.org/wiki/Silhouette_(clustering)), but it is possible to explicitly specify the desired number of clusters, e.g., `./cluster 1`.

The training procedure is implemented in the `./train.py` script. It accepts various parameters:

* `--model=MODEL`, where `MODEL` is the desired model,
* `--gpu=1` that suggests the program to use a GPU, when possible,
* `--num_epochs=14000` that specifies the number of training epochs,
* `--batch_size=512` that specifies the batch size.

After the training, the number of `MODEL.W-k.txt` files being generated representing the projection matrix for each `k` cluster.

## Evaluating

The evaluation script has only one parameter: the previously trained model to evaluate. Example: `./evaluate.py baseline`. It is also possible to study how good (but usually bad) the original embeddings represent the subsumptions. For that, it is simply enough to run `./identity.py`.

## Copyright

Copyright (c) 2016 [Dmitry Ustalov](https://ustalov.name/en/). See LICENSE for details.
