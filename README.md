# Learning Word Subsumption Projections

## Prerequisites

This implementation uses [TensorFlow](https://www.tensorflow.org/).

The following datasets are required to run these software:

* the semantic relations: [projlearn-ruwikt.tar.gz]
* the trained word2vec model: [all.norm-sz100-w10-cb0-it1-min100.w2v].

[projlearn-ruwikt.tar.gz]: http://ustalov.imm.uran.ru/pub/projlearn-ruwikt.tar.gz
[all.norm-sz100-w10-cb0-it1-min100.w2v]: https://s3-eu-west-1.amazonaws.com/dsl-research/wiki/w2v_export/all.norm-sz100-w10-cb0-it1-min100.w2v

## Training

* `./prepare.py`
* `./cluster.py`
* `./train.py`

## Evaluating

* `./evaluate.py`
* `./identity.py`

## Copyright

Copyright (c) 2016 [Dmitry Ustalov](https://ustalov.name/en/). See LICENSE for details.
