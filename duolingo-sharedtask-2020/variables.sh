# this doesn't change.
src=en
tgt=ja

lang=${src}-${tgt}
DATA=data/courses/${lang}
DUMP=data/dumps/${lang}

# download these first.
# git clone https://github.com/moses-smt/mosesdecoder
# git clone https://github.com/rsennrich/subword-nmt
MOSES=$PWD/mosesdecoder
SUBWORDNMT=$PWD/subword-nmt


# Location of the shared task data.
SHARED_TASK_DATA=$PWD/staple-2020-train

# used in run_pretrained.sh (don't change NBEST)
NBEST=8

CANDLIMIT=129
