#!/bin/bash -ex

export LANG=en_US.utf8

# Just a prefix for the convenience.
PREFIX=sz100

# The number of training epochs.
NUM_EPOCHS=700

# The batch size.
BATCH_SIZE=1024

# The random seed.
SEED=228

# The word2vec model.
W2V=all.norm-sz100-w10-cb0-it1-min100.w2v

# The cluster sizes to iterate over.
CLUSTERS=$(seq 1 8)

# The lambda values to iterate over.
LAMBDAS=$(seq 0.1 0.1 0.3)

for k in $CLUSTERS; do
  ./cluster.py -k $k

  for model in baseline; do
    CWD=$PREFIX-k$k-l0.0
    mkdir -pv $CWD

    ./train.py --model=$model --num_epochs=$NUM_EPOCHS --batch_size=$BATCH_SIZE --seed=$SEED --test=validation.npz | tee $model-train.log
    mv -f $model-train.log $model.*.trained* $model.test.npz $CWD/
    cp kmeans.pickle $CWD/

    CWDS="$CWDS $CWD"
  done

  for lambda in $LAMBDAS; do
    CWD=$PREFIX-k$k-l$lambda
    mkdir -pv $CWD

    for model in regularized_hyponym regularized_synonym; do
      ./train.py --model=$model --num_epochs=$NUM_EPOCHS --batch_size=$BATCH_SIZE --seed=$SEED --lambdac=$lambda --test=validation.npz | tee $model-train.log
      mv -f $model-train.log $model.*.trained* $model.test.npz $CWD/
      cp kmeans.pickle $CWD/
    done

    CWDS="$CWDS $CWD"
  done

  ./evaluate.py --w2v=$W2V --test=validation.npz --subsumptions=subsumptions-validation.txt $CWDS | tee -a $PREFIX-validation.log
  unset CWDS
done
