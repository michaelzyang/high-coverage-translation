# STAPLE 2020 README

Welcome to the Duolingo 2020 Shared Task!

## Data

There are five folders, one for each language. Each folder contains two files: 1) the shared task data (example, test.en_hu.2020-02-20.prompts.txt) and 2) the AWS translation baseline (example, test.en_hu.aws_baseline.pred.txt).

In order to score this data, you can submit to our CodaLab!

## Code
Several code stubs have been provided, including code to train baseline models with fairseq.

* `variables.sh` : common BASH variables
* `preprocess.sh` : to preprocess the data for training with fairseq
* `train.sh` : to train the model using preprocessed data
* `run_pretrained.sh` : script to run pretrained fairseq models (also provided)
* `utils.py` : utility functions, you may also find them useful
* `staple-2020-scorer.py` : official scoring script

Python libraries used: fairseq, sacremoses, subword_nmt, fairseq, sacrebleu, tqdm

Good luck!

If you have questions, feel free to check or post to the mailing list: https://groups.google.com/forum/#!forum/duolingo-sharedtask-2020
