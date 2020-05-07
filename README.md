# High Coverage Translation System for the ACL WNGT STAPLE 2020 Shared Task
This repository contains the code used to train and evaluate our High Coverage Translation System.

## Training
The pretrained model we worked from was the JParaCrawl English-to-Japanese Transformer 'base' model (and associated sentencepiece models) from http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/. 

Model finetuning was conducted in the `jparacrawl-finetune` submodule, branched from https://github.com/MorinoseiMorizo/jparacrawl-finetune
The shell scripts for training our various models were the `jparacrawl-finetune/fine-tune_preprocess.sh` and `jparacrawl-finetune/staple_*.sh` scripts. These were adapted from the original scripts in the `jparacrawl-finetune` repository.

Models and experiments for the neural filtering model is in `filtering` folder.

## Corpora
Our Duolingo train/dev/test gold data can be found at `duolingo-sharedtask-2020/staple-2020-train/en_ja/*_split.gold.txt`. We created these split files from the original `train.en_ja.2020-01-13.gold.txt` provided for the shared task (excluded from the repo due to excessive file size) https://sharedtask.duolingo.com/. 

We used the 'official splits' JESC data from https://nlp.stanford.edu/projects/jesc/.

Before training or evaluation, the raw sentences in these files were encoded into subword sequences using the aforementioned sentencepiece models from JParaCrawl.

## Evaluation
We adapted the scoring pipeline provided in the shared task starter code https://github.com/duolingo/duolingo-sharedtask-2020/.
We edited `duolingo-sharedtask-2020/my_cands_extract_spm.py` to decode our subword translation outputs using our sentencepiece model.
We edited `duolingo-sharedtask-2020/staple_2020_scorer.py` to print descriptive statistics for the number of candidates outputed per prompt.
We created `duolingo-sharedtask-2020/qualitative_analysis.py` to conduct qualitative error analysis between the outputs of different models.
