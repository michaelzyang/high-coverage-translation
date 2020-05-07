#!/bin/bash

set -e


# get the parent directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/variables.sh

OUTPUT=$DATA/output
mkdir -p $OUTPUT







# Set this to your favorite model!
MODEL=../jparacrawl-finetune/models/staple-jesc-duo-weighted-0-05/average/average.pt

# PUT THE FILE WITH THE DESIRED PROMPTS HERE - MAKE SURE YOU SET THE RIGHT FOLDER AND FILE FOR TESTFILE AND fairseq-generate --gen-subset
#TESTFILE=${SHARED_TASK_DATA}/${src}_${tgt}/train.${src}_${tgt}.2020-01-13.gold.txt
#TESTFILE=./staple-2020-dev-blind/${src}_${tgt}/dev.${src}_${tgt}.2020-02-20.prompts.sep.txt
#TESTFILE=./staple-2020-test-blind/${src}_${tgt}/test.${src}_${tgt}.2020-02-20.prompts.sep.txt
TESTFILE=${SHARED_TASK_DATA}/${src}_${tgt}/test_split.gold.txt

# FAIRSEQDIR=data/courses/en-${tgt}/raw_splits/
FAIRSEQDIR=../jparacrawl-finetune/duolingo/raw_splits_weighted/

# DON'T FORGET TO SELECT THE RIGHT --gen-subset {train, valid, test}
fairseq-generate $FAIRSEQDIR --path $MODEL --dataset-impl raw --raw-text \
   --gen-subset test \
   --max-tokens 200 \
   --diverse-beam-groups 1 \
   --diverse-beam-strength 0.1 \
   --beam $NBEST --batch-size 128 --remove-bpe --nbest $NBEST --replace-unk > $OUTPUT/gen.out

# THIS SHOULD BE my_cands_extract_train.py or my_cands_extract_dev_test.py IF USING BPE VOCAB OR my_cands_extract_spm.py IF USING SENTENCEPIECE
EXTRACT=my_cands_extract_spm.py
 






# this cleans all the BPE
#sed -i '' 's/@@ //g' $OUTPUT/gen.out

python3 $EXTRACT --origfile $TESTFILE --infile $OUTPUT/gen.out --outfile $OUTPUT/all_cands.txt --candlimit $CANDLIMIT
mv sys.out ref.out ${OUTPUT}

cat ${OUTPUT}/all_cands.txt | $MOSES/scripts/tokenizer/detokenizer.perl -l $tgt > ${OUTPUT}/all_cands_detok.txt

# EXIT BEFORE SCORING IF USING BLIND DEV OR TEST
# exit 0



# who creates all_cands.txt? my_cands_extract.py
python3 staple_2020_scorer.py --gold $TESTFILE --pred ${OUTPUT}/all_cands_detok.txt

exit 0
# this will compute BLEU score (best translation vs the top hypothesis)
#fairseq-score --sys ${OUTPUT}/sys.out --ref ${OUTPUT}/ref.out
cat ${OUTPUT}/sys.out | $MOSES/scripts/tokenizer/detokenizer.perl > sys_detok
cat ${OUTPUT}/ref.out | $MOSES/scripts/tokenizer/detokenizer.perl > ref_detok

cat sys_detok | sacrebleu -lc ref_detok
rm sys_detok ref_detok
