#!/bin/bash
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

FAIRSEQ=/workspace/fairseq

SEED=1

EXP_NAME=staple-jesc-duo-weighted

SRC=en
TRG=ja

PRETRAINED_MODEL_FILE=$PWD/models/staple-jesc/average/average.pt

SPM_MODEL=$PWD/duolingo/enja_spm_models/spm.$TRG.nopretok.model

DATA_DIR=$PWD/duolingo/bin_splits_weighted
TEST_TRG_RAW=$PWD/duolingo/raw_splits_weighted/test-split-sents.$TRG

MODEL_DIR=$PWD/models/$EXP_NAME
mkdir -p $MODEL_DIR


######################################
# Training
######################################
python3 $FAIRSEQ/train.py $DATA_DIR \
    --restore-file $PRETRAINED_MODEL_FILE \
    --arch transformer \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 1.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-7 \
    --warmup-updates 4000 \
    --lr 0.001 \
    --min-lr 1e-09 \
    --dropout 0.3 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 5000 \
    --max-update 28000 \
    --save-dir $MODEL_DIR \
    --no-epoch-checkpoints \
    --save-interval 10000000000 \
    --validate-interval 1000000000 \
    --save-interval-updates 100 \
    --keep-interval-updates 8 \
    --log-format simple \
    --log-interval 5 \
    --ddp-backend no_c10d \
    --update-freq 16 \
    --fp16 \
    --seed $SEED \
    --dataset-impl mmap

######################################
# Averaging
######################################
rm -rf $MODEL_DIR/average
mkdir -p $MODEL_DIR/average
python3 $FAIRSEQ/scripts/average_checkpoints.py --inputs $MODEL_DIR --output $MODEL_DIR/average/average.pt --num-update-checkpoints 8


######################################
# Generate
######################################
# decode
B=`basename $TEST_SRC`

python3 $FAIRSEQ/generate.py $DATA_DIR \
    --gen-subset test \
    --path $MODEL_DIR/average/average.pt \
    --max-tokens 1000 \
    --beam 6 \
    --lenpen 1.0 \
    --log-format simple \
    --remove-bpe \
    > $MODEL_DIR/$B.hyp

grep "^H" $MODEL_DIR/$B.hyp | sed 's/^H-//g' | sort -n | cut -f3 > $MODEL_DIR/$B.true
spm_decode --model=$SPM_MODEL --input_format=piece < $MODEL_DIR/$B.true > $MODEL_DIR/$B.true.detok


######################################
# Evaluation
######################################
# mecab tokenize
mecab -Owakati < $TEST_TRG_RAW > $MODEL_DIR/$B.mecab.ref
mecab -Owakati < $MODEL_DIR/$B.true.detok > $MODEL_DIR/$B.mecab.hyp
cat $MODEL_DIR/$B.mecab.hyp | sacrebleu --tokenize=intl $MODEL_DIR/$B.mecab.ref | tee -a $MODEL_DIR/test.log
