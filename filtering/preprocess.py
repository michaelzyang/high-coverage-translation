from utils import FIELDSEP, read_trans_prompts, read_transfile
import json
import sentencepiece as spm

SPM_PATH = 'data/courses/en-ja/bin/enja_spm_models/spm.ja.nopretok.model' # path to sentencepiece model
sp = spm.SentencePieceProcessor()
sp.Load(SPM_PATH)

SPM_PATH = 'data/courses/en-ja/bin/enja_spm_models/spm.en.nopretok.model' # path to sentencepiece model
en_sp = spm.SentencePieceProcessor()
en_sp.Load(SPM_PATH)

def preprocess(split):
    """ preprocess data for filtering model """
    origfile = "all_cands_%s.txt"%split
    with open(origfile) as f:
        orig_prompts = read_transfile(f.readlines(), weighted=False) 
    with open(origfile) as f:
        src_lines = read_trans_prompts(f.readlines())
    src_lines = {a:b for (a, b) in src_lines}
    data = []
    for (key, x) in orig_prompts.items():
        candidates = []
        for k in x.keys():
            score = 0
            sent = sp.EncodeAsPieces(k)
            candidates.append((sent, score))
        candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
        candidates = [{"id": i, "tokens": x[0], "score": x[1]} for (i, x) in enumerate(candidates)]
        data.append({"src": key, "cand": candidates})
    cand = data
    origfile = "%s_split.gold.txt"%split
    with open(origfile) as f:
        orig_prompts = read_transfile(f.readlines(), weighted=True) 
    data = []
    for (key, x) in orig_prompts.items():
        candidates = []
        for k in x.keys():
            score = 0
            sent = sp.EncodeAsPieces(k)
            candidates.append((sent, score))
        candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
        candidates = [{"id": i, "tokens": x[0], "score": x[1]} for (i, x) in enumerate(candidates)]
        data.append({"src": key, "cand": candidates})
    gold = data
    print(len(gold))
    gold = {x["src"]:x["cand"] for x in gold}
    cand = {x["src"]:x["cand"] for x in cand}
    data = []
    for (k, v) in gold.items():
        x = cand[k]
        real = set([" ".join(y["tokens"]) for y in v])
        num = len(x)
        cnt = 0
        for item in x:
            if " ".join(item["tokens"]) in real:
                item["score"] = 1
                cnt += 1
        src_tokens = en_sp.EncodeAsPieces(src_lines[k])
        data.append({"src": src_tokens, "cand": x})
    with open("./data/%s.ja.rank.aug.json"%split, "w") as f:
        json.dump(data, f)

def split(name):
    """ split NMT results """
    with open("%s_split.gold.txt"%name) as f:
        lines = f.readlines()
    keys = []
    for x in lines:
        if x.startswith("prompt"):
            keys.append(x)
    print(len(keys))
    keys = set(keys)
    with open("all_cands_detok.txt") as f:
        lines = f.readlines()
    cnt = 0
    with open("all_cands_%s.txt"%name, "w") as f:
        flag = True
        cand = set()
        for x in lines:
            if x.startswith("prompt"):
                if x in keys:
                    cand = set()
                    flag = True
                    cnt += 1
                else:
                    flag = False
            if flag:
                if x.startswith("prompt"):
                    f.write(x)
                elif x not in cand:
                    f.write(x)
                    cand.add(x)
    print(cnt)

if __name__ == "__main__":
    split("train")
    split("test")
    split("dev")
    preprocess("train")
    preprocess("test")
    preprocess("dev")