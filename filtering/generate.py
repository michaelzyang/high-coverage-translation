import torch
import torch.nn as nn
import torch.optim as optim
import json
from models import BinaryModel
from dictionary import Dictionary
import numpy as np
import argparse
from main import base_architecture
import sentencepiece as spm
from dataloader import pad_seq
from staple_2020_scorer import test
import random

SPM_PATH = 'data/courses/en-ja/bin/enja_spm_models/spm.ja.nopretok.model' # path to sentencepiece model 
tgt_sp = spm.SentencePieceProcessor()
tgt_sp.Load(SPM_PATH)

SPM_PATH = 'data/courses/en-ja/bin/enja_spm_models/spm.en.nopretok.model' # path to sentencepiece model
src_sp = spm.SentencePieceProcessor()
src_sp.Load(SPM_PATH)

def process_file(fname, weighted):
    """ process results from NMT model """
    with open(fname) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    first = True
    data = []
    for x in lines:
        if len(x) == 0:
            continue
        elif x.startswith("prompt"):
            item = {"source": x, "cand": []}
            data.append(item)
        else:
            last_id = len(data) - 1
            if weighted:
                data[last_id]["cand"].append(x.split("|")[0])
            else:
                data[last_id]["cand"].append(x)
    return data

def write_file(result, fname):
    """ write results """
    with open(fname, "w") as f:
        for x in result:
            print(file=f)
            print(x["source"], file=f)
            for c in x["cand"]:
                print(c, file=f)

def batchify(sents, sp, dictionary, cuda):
    """ batch sentences """
    sents = [sp.EncodeAsPieces(x) for x in sents]
    sents = [[dictionary.word2idx[x] for x in s] for s in sents]
    sents = pad_seq([torch.LongTensor(x) for x in sents], dictionary.pad_token_id)
    if cuda:
        sents = sents.cuda()
    return sents
    
def rerank_binary(args):
    """ generate and evaluate results """
    print("rerank binary")
    base_architecture(args)
    data = process_file("all_cands_%s.txt"%args.split, args.weighted)
    tgt_dictionary = Dictionary.load("./data/ja.dict")
    src_dictionary = Dictionary.load("./data/en.dict")
    model = BinaryModel(len(src_dictionary), args.word_dim, len(tgt_dictionary), args.word_dim, args.hidden_dim, args.dropout)
    if args.cuda:
        model = model.cuda()
    model.load_state_dict(torch.load(args.model_pt))
    model.eval()
    result = []
    print("generate")
    with torch.no_grad():
        # iterate througe samples
        for (i, x) in enumerate(data):
            if i % 500 == 0:
                print(i)
            sents = batchify(x["cand"], tgt_sp, tgt_dictionary, args.cuda)
            src = batchify([x["source"].split("|")[1]], src_sp, src_dictionary, args.cuda)
            pred_scores = model(src, sents)
            pred_scores = torch.sigmoid(pred_scores)
            pred_scores = pred_scores.squeeze().cpu().numpy().tolist()
            pred = zip(pred_scores, x["cand"])
            pred_scores, cands = zip(*pred)
            reranked = []
            for (i, p) in enumerate(pred_scores):
                # filtering
                if p > args.prob:
                    reranked.append(cands[i])
            reranked = reranked + list(set(cands[:10]).difference(set(reranked)))
            reranked = reranked[:args.max_cand]
            result.append({"source": x["source"], "cand": reranked})
    write_file(result, args.result_pt)
    print("evaluate")
    test("%s_split.gold.txt"%args.split, args.result_pt)

def random_oracle(args):
    """ random oracle """
    data = process_file(args.fname, args.weighted)
    dictionary = Dictionary.load("./data/ja.dict")
    result = []
    print("generate")
    for (i, x) in enumerate(data):
        if i % 500 == 0:
            print(i)
        sents = x["cand"]
        reranked = sents[:args.max_cand]
        result.append({"source": x["source"], "cand": reranked})
    write_file(result, args.result_pt)
    print("evaluate")
    test("./staple-2020-train/en_ja/train.en_ja.2020-01-13.gold.txt", args.result_pt)

def oracle(args):
    """ oracle which selects the first K candidates """
    data = process_file(args.fname, args.weighted)
    data = data[:2100]
    ref = process_file("./data/train.gold.txt", True)
    dictionary = Dictionary.load("./data/ja.dict")
    result = []
    print("generate")
    for (i, x) in enumerate(data):
        if i % 500 == 0:
            print(i)
        sents = x["cand"]
        ref_sents = set(ref[i]["cand"])
        reranked = []
        for s in sents:
            if s in ref_sents:
                reranked.append(s)
        if len(reranked) == 0:
            reranked.append(x["cand"][0])
        result.append({"source": x["source"], "cand": reranked})
    write_file(result, args.result_pt)
    test("./data/train.gold.txt", args.result_pt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Parameter')
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument("--max_cand", default=100, type=int)
    parser.add_argument("--fname", default="all_cands_detok_200.txt", type=str)
    parser.add_argument("--model_pt", default="./cache/model.pt", type=str)
    parser.add_argument("--result_pt", default="cands_rerank.txt", type=str)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--split", default="dev", type=str)
    parser.add_argument("-p", "--prob", default=0, type=float)
    args = parser.parse_args()
    if args.random:
        print("random")
        random_oracle(args)
    else:
        print("rerank")
        if args.cuda:
            with torch.cuda.device(args.gpuid):
                rerank_binary(args)
        else:
            rerank_binary(args)
