# Hyperstar: Negative Sampling Improves Hypernymy Extraction Based on Projection Learning

We present a new approach to the extraction of hypernyms based on projection learning and word embeddings. In contrast to classification-based approaches, projection-based methods require no candidate hyponym-hypernym pairs. While it is natural to use both positive and negative training examples in supervised relation extraction, the impact of negative examples on hypernym prediction was not studied so far. In this paper, we show that explicit negative examples used for regularization of the model significantly improve performance compared to the state-of-the-art approach of Fu et al. ([2014](https://doi.org/10.3115/v1/P14-1113)) on three datasets from different languages.

This repository contains the implementation of our approach, called Hyperstar. The dataset produced in our study is available on Zenodo for both English and Russian.

[![Paper][paper_badge]][paper_link] [![Docker Hub][docker_badge]][docker_link] [![Dataset][zenodo_badge]][zenodo_link]

[paper_badge]: https://img.shields.io/badge/EACL%202017-10.18653%2Fv1%2FE17--2087-success
[paper_link]: https://doi.org/10.18653/v1/E17-2087
[docker_badge]: https://img.shields.io/docker/pulls/nlpub/hyperstar.svg
[docker_link]: https://hub.docker.com/r/nlpub/hyperstar/
[zenodo_badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.290524.svg
[zenodo_link]: https://doi.org/10.5281/zenodo.290524

## Citation

In case this software, the study, or the dataset was useful for you, please cite our EACL&nbsp;2017 paper.

* [Ustalov, D.](https://github.com/dustalov), [Arefyev, N.](https://github.com/nvanva), [Biemann, C.](https://www.inf.uni-hamburg.de/en/inst/ab/lt/people/chris-biemann.html), [Panchenko, A.](https://www.inf.uni-hamburg.de/en/inst/ab/lt/people/alexander-panchenko.html): [Negative Sampling Improves Hypernymy Extraction Based on Projection Learning](https://doi.org/10.18653/v1/E17-2087). In: Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume&nbsp;2, Short Papers, Valencia, Spain, Association for Computational Linguistics (April 2017) 543&ndash;550

```bibtex
@inproceedings{Ustalov:17:eacl,
  author    = {Ustalov, Dmitry and Arefyev, Nikolay and Biemann, Chris and Panchenko, Alexander},
  title     = {{Negative Sampling Improves Hypernymy Extraction Based on Projection Learning}},
  booktitle = {Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume~2, Short Papers},
  series    = {EACL~2017},
  month     = {April},
  year      = {2017},
  address   = {Valencia, Spain},
  publisher = {Association for Computational Linguistics},
  pages     = {543--550},
  isbn      = {978-1-945626-35-7},
  doi       = {10.18653/v1/E17-2087},
  language  = {english},
}
```

## Reproducibility

To reproduce our experimental results, you need dictionaries ([LexNet](https://github.com/vered1986/LexNET/tree/v2/datasets) for English, [Parsed Wiktionary](http://ustalov.imm.uran.ru/pub/projlearn-ruwikt.tar.gz) for Russian) and word embeddings ([Google News](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) for English, [Russian Distributional Thesaurus](http://panchenko.me/data/dsl-backup/w2v-ru/all.norm-sz500-w10-cb0-it3-min5.w2v) for Russian). Since our implementation uses Python&nbsp;3 and TensorFlow&nbsp;0.12, please install them, too.

```shell
python3 -m venv venv
./venv/bin/pip3 -r requirements.txt
```

We prepared the Docker image [nlpub/hyperstar](https://hub.docker.com/r/nlpub/hyperstar/) that contains the necessary dependencies for running our software. However, we recommend using a virtualenv instead.

Please make sure you specified the correct word embedding model in every invocation of Hyperstar scripts.

### Preparation

The input dictionaries should be transformed into the format used by Hyperstar. Words for which there is no embeddings should be excluded. This is achieved by running the `./dictionary.en.py` script for English and `./dictionary.ru.py` for Russian. Then, the word embeddings should be dumped for the further processing using the `./prepare.py` script. These scripts might take significant amount of time, but they are executed only once. Finally, the vector space should be separated into a number of clusters using the `./cluster.py -k 1` script, where an arbitrary number of clusters can be specified instead of `1`. This is found to be very useful for improving the results, so it is not possible to proceed without clustering.

### Training

The original approach by Fu et al. (2014) learns a matrix that transforms an input hyponym vector into its hypernym vector. This approach is implemented as a `baseline`, while Hyperstar features various regularized approaches:

* `baseline`, the original approach
* `regularized_hyponym` that penalizes the matrix for transforming the hypernyms back to the hyponyms
* `regularized_synonym` that penalizes the matrix for transforming the hypernyms back to the synonyms of the hyponyms
* `regularized_hypernym` that promotes the matrix for transforming the hyponym synonyms to the hypernyms

The training script, `./train.py`, accepts the following parameters:

* `--model=MODEL`, where `MODEL` is the desired approach described above
* `--gpu=1` that suggests the program to use a GPU, when possible
* `--num_epochs=300` that specifies the number of training epochs
* `--batch_size=2048` that specifies the batch size
* `--stddev=0.01` that specifies the standard deviation for initializing the transformation matrices
* `--lambdac=0.10` that specifies the regularization coefficient

The trained models are written to `MODEL.k%d.trained` files. Each file represents the trained model for each cluster. The data for further evaluation are written into the `MODEL.test.npz` file.

### Evaluation

The evaluation script requires the previously trained model: `./evaluate.py path-to-the-trained-model`. It is also possible to study how good (but usually bad) the intact embeddings represent the subsumptions by running `./identity.py`.

It is possible to reuse our post-processing scripts for parameter tuning (`./enumerate.sh`), evaluation log parsing (`./parse-logs.awk sz100-validation.log >sz100-validation.tsv`), and data visualization (`R --no-save <evaluate.R`).

## Copyright

Copyright (c) 2016&ndash;2017 [Dmitry Ustalov](https://github.com/dustalov) and others. See LICENSE for details.
