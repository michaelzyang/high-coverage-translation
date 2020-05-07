#!/bin/bash
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

FAIRSEQ=/workspace/fairseq

SEED=1

SRC=en
TRG=ja

CORPUS_DIR=$PWD/duolingo/raw_splits_weighted
DATA_DIR=$PWD/duolingo/bin_splits_weighted

TRAIN_PREFIX=$CORPUS_DIR/train.${SRC}-${TRG}
DEV_PREFIX=$CORPUS_DIR/valid.${SRC}-${TRG}
TEST_PREFIX=$CORPUS_DIR/test.${SRC}-${TRG}
SRC_VOCAB=$PWD/pretrained_model_$SRC$TRG/dict.$SRC.txt
TRG_VOCAB=$PWD/pretrained_model_$SRC$TRG/dict.$TRG.txt

mkdir $DATA_DIR

######################################
# Preprocessing
######################################
python3 $FAIRSEQ/preprocess.py \
    --source-lang $SRC \
    --target-lang $TRG \
    --trainpref $TRAIN_PREFIX \
    --validpref $DEV_PREFIX \
    --testpref $TEST_PREFIX \
    --destdir $DATA_DIR \
    --srcdict $SRC_VOCAB \
    --tgtdict $TRG_VOCAB \
    --workers `nproc` \

