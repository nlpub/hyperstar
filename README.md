# Learning Word Subsumption Projections

This is the implementation of the projection learning approach for learning word subsumptions, i.e., hyponyms and hypernyms, originally proposed by Fu et al. ([2014](http://dx.doi.org/10.3115/v1/P14-1113)). The approach requires pre-trained word embeddings in the [word2vec](https://code.google.com/archive/p/word2vec/) format and the list of subsumption examples to learn the projection matrix. This implementation uses [TensorFlow](https://www.tensorflow.org/).

## Citation

In case this software, the study or the dataset was useful for you, please cite the following paper.

* [Ustalov, D.](https://github.com/dustalov), [Arefyev, N.](https://github.com/nvanva), [Biemann, C.](https://www.inf.uni-hamburg.de/en/inst/ab/lt/people/chris-biemann.html), [Panchenko, A.](https://www.inf.uni-hamburg.de/en/inst/ab/lt/people/alexander-panchenko.html): [Negative Sampling Improves Hypernymy Extraction Based on Projection Learning](https://doi.org/10.18653/v1/E17-2087). In: Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers, Valencia, Spain, Association for Computational Linguistics (April 2017) 543–550

```latex
@inproceedings{Ustalov:17:eacl,
  author    = {Ustalov, Dmitry and Arefyev, Nikolay and Biemann, Chris and Panchenko, Alexander},
  title     = {{Negative Sampling Improves Hypernymy Extraction Based on Projection Learning}},
  booktitle = {Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers},
  month     = {April},
  year      = {2017},
  address   = {Valencia, Spain},
  publisher = {Association for Computational Linguistics},
  pages     = {543--550},
  isbn      = {978-1-945626-35-7},
  doi       = {10.18653/v1/E17-2087},
  url       = {http://www.aclweb.org/anthology/E17-2087},
  language  = {english},
}
```

## Reproducibility

We prepared the Docker image [nlpub/hyperstar](https://hub.docker.com/r/nlpub/hyperstar/) that contains the necessary dependencies for running our software. Also, the datasets produced in the research paper mentioned above are published on ZENODO: <https://doi.org/10.5281/zenodo.290524>.

[![Docker Hub][docker_badge]][docker_link] [![ZENODO][zenodo_badge]][zenodo_link]

[docker_badge]: https://img.shields.io/docker/pulls/nlpub/hyperstar.svg
[docker_link]: https://hub.docker.com/r/nlpub/hyperstar/
[zenodo_badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.290524.svg
[zenodo_link]: https://doi.org/10.5281/zenodo.290524

## Prerequisites

This implementation is designed for processing the Russian language, but there should be no problem in running it on any other language provided with the relevant datasets. However, for processing the Russian language, the following datasets are required:

* the trained word2vec model: [all.norm-sz100-w10-cb0-it1-min100.w2v],
* the set of semantic relations: [projlearn-ruwikt.tar.gz].

[projlearn-ruwikt.tar.gz]: http://ustalov.imm.uran.ru/pub/projlearn-ruwikt.tar.gz
[all.norm-sz100-w10-cb0-it1-min100.w2v]: http://panchenko.me/data/dsl-backup/w2v-ru/all.norm-sz100-w10-cb0-it1-min100.w2v

## Models

The original approach learns a matrix such that transforms an input hyponym embedding vector into its hypernym embedding vector. A few variations featuring additive regularization of this approach have also been implemented. The following models are available:

* `baseline`, the original approach,
* `regularized_hyponym` that penalizes the matrix to projecting the hypernyms back to the hyponyms,
* `regularized_synonym` that penalizes the matrix to projecting the hypernyms back to the synonyms of the hyponyms,
* `regularized_hypernym` that promotes the matrix to projecting the hyponym synonyms to the hypernyms,
* `frobenius_loss` that uses the [Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm) as the loss function for `baseline`.

## Training

Before any processing, certain things need to be precomputed, such as training and test sets, etc. For that, the `./dictionary.ru.py` and `./prepare.py` scripts should be executed. On large embeddings, it might take a long time, but it is run only once.

Having the preparation script finished, the vector space should be separated into a number of clusters using the `./cluster.py` script. This is found to be very useful to improving the results, so it is not possible to continue without clustering. Usually, the clustering program automatically estimates the number of clusters using the [silhouette method](https://en.wikipedia.org/wiki/Silhouette_(clustering)), but it is possible to explicitly specify the desired number of clusters, e.g., `./cluster.py -k 1`.

The training procedure is implemented in the `./train.py` script. It accepts different parameters:

* `--model=MODEL`, where `MODEL` is the desired model,
* `--gpu=1` that suggests the program to use a GPU, when possible,
* `--num_epochs=300` that specifies the number of training epochs,
* `--batch_size=2048` that specifies the batch size,
* `--stddev=0.01` that specifies the standard deviation for initializating the projection matrix,
* `--lambdac=0.10` that specifies the regularization coefficient.

After the training, the number of `MODEL.k%d.trained` files being generated representing the trained model for each cluster. Also, the data for evaluation are written into the `MODEL.test.npz` file.

## Evaluating

The evaluation script has only one parameter: the previously trained model to evaluate. Example: `./evaluate.py path-with-the-trained-model`. It is also possible to study how good (but usually bad) the original embeddings represent the subsumptions. For that, it is simply enough to run `./identity.py`.

## Copyright

Copyright (c) 2016&ndash;2017 [Dmitry Ustalov](https://ustalov.name/en/) and others. See LICENSE for details.
