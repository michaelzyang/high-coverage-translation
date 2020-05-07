"""Prepares gold prompts out of the blind dev and test files (for my_cands_extract.py): puts '\n' between every line"""
import argparse

def separate_prompt(in_name, out_name):
    with open(in_name, 'r') as in_f, open(out_name, "w") as out_f:
        lines = in_f.readlines()
        out_f.write('\n'.join(lines))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("This converts data in the shared task format into standard machine translation format (one sentence per line, languages in separate files.)")
    parser.add_argument("--infile", help="Path of shared task file (probably something like train.en_vi.2020-01-13.gold.txt)", required=True)
    parser.add_argument("--outfile", help="Name of desired src file, probably something like train_sents.en", required=True)
    args = parser.parse_args()

    separate_prompt(args.infile, args.outfile)