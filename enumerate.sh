#!/bin/bash -ex

# Just a prefix for the convenience.
PREFIX=sz100

# The number of training epochs.
NUM_EPOCHS=700

# The batch size.
BATCH_SIZE=1024

# The random seed.
SEED=228

for k in 1 2 3 4 5 6 7 8; do
  ./cluster.py -k $k

  for model in baseline; do
    CWD=$PREFIX-k$k-l0.0
    mkdir -pv $CWD

    ./train.py --nogpu --model=$model --num_epochs=$NUM_EPOCHS --batch_size=$BATCH_SIZE --seed=$SEED --test=validation.npz | tee $model-train.log
    mv -f *-train.log *.trained* $CWD/
    cp kmeans.pickle $CWD/

    ./evaluate.py --test=validation.npz --subsumptions=subsumptions-validation.txt $CWD | tee -a $PREFIX-validation.log
  done

  for lambda in 0.1 0.2 0.3; do
    CWD=$PREFIX-k$k-l$lambda
    mkdir -pv $CWD

    for model in regularized_hyponym regularized_synonym; do
      ./train.py --nogpu --model=$model --num_epochs=$NUM_EPOCHS --batch_size=$BATCH_SIZE --seed=$SEED --lambdac=$lambda --test=validation.npz | tee $model-train.log
      mv -f *-train.log *.trained* $CWD/
      cp kmeans.pickle $CWD/
    done

    CWDS="$CWDS $CWD"
  done

  ./evaluate.py --test=validation.npz --subsumptions=subsumptions-validation.txt $CWDS | tee -a $PREFIX-validation.log
  unset CWDS
done
