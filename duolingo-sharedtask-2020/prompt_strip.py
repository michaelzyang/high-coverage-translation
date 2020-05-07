"""Makes input sents for blind dev and test files: strips the text to the left of the | on every line"""
import argparse

def strip_prompt(in_name, out_name):
    with open(in_name, 'r') as in_f, open(out_name, "w") as out_f:
        lines = in_f.readlines()
        out_lines = [l.split('|')[-1] for l in lines]
        out_f.write(''.join(out_lines))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("This converts data in the shared task format into standard machine translation format (one sentence per line, languages in separate files.)")
    parser.add_argument("--infile", help="Path of shared task file (probably something like train.en_vi.2020-01-13.gold.txt)", required=True)
    parser.add_argument("--outfile", help="Name of desired src file, probably something like train_sents.en", required=True)
    args = parser.parse_args()

    strip_prompt(args.infile, args.outfile)