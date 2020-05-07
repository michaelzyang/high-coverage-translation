import argparse
from utils import read_transfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goldfile", help="gold file", required=True)
    parser.add_argument("--predfile1", help="pred file 1", required=True)
    parser.add_argument("--predfile2", help="pred file 2", required=True)
    args = parser.parse_args()

    with open(args.goldfile) as f:
        print("reading gold")
        gold = read_transfile(f.readlines(), weighted=True)

    with open(args.predfile1) as f:
        print("reading pred1")
        pred1 = read_transfile(f.readlines())

    with open(args.predfile2) as f:
        print("reading pred2")
        pred2 = read_transfile(f.readlines())

    # Choose a prompt
    keys = list(gold.keys())
    key = keys[0]

    # Get the translations for that prompt
    gold_set = set(gold[key].keys())
    pred1_keys = set(pred1[key].keys())
    pred2_keys = set(pred2[key].keys())
    print(f'Pred1 {len(gold_set.intersection(pred1_keys))} / {len(pred1_keys)} correct')
    print(f'Pred2 {len(gold_set.intersection(pred2_keys))} / {len(pred2_keys)} correct')

    # Get the translations unique to each set
    pred1_only = pred1_keys - pred2_keys
    pred2_only = pred2_keys - pred1_keys
    both_right = gold_set.intersection(pred1_keys).intersection(pred2_keys)
    pred1_only_right = [x for x in pred1_only if x in gold_set]
    pred1_only_wrong = [x for x in pred1_only if x not in gold_set]
    pred2_only_right = [x for x in pred2_only if x in gold_set]
    pred2_only_wrong = [x for x in pred2_only if x not in gold_set]

    # Print them
    def print_each(li, header):
        print(header)
        for x in li:
            print(x)


    print_each(both_right, 'both right')
    print_each(pred1_only_right, 'pred1_only_right')
    print_each(pred2_only_right, 'pred2_only_right')
    print_each(pred1_only_wrong, 'pred1_only_wrong')
    print_each(pred2_only_wrong, 'pred2_only_wrong')

