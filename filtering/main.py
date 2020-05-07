import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import MatchLoader
import json
from models import BinaryModel
from dictionary import Dictionary
import numpy as np
import argparse

def base_architecture(args):
    """ hyperparameters for model architecture """
    args.word_dim = getattr(args, "word_dim", 128)
    args.hidden_dim = getattr(args, "hidden_dim", 128)
    args.dropout = getattr(args, "dropout", 0)
    args.lr = getattr(args, "lr", 1e-4)
    args.grad_clip = getattr(args, "grad_clip", 1)

def main(args):
    """ main process """
    base_architecture(args)
    # load data
    with open("./data/train.ja.rank.aug.json") as f:
        train_data = json.load(f)
    with open("./data/dev.ja.rank.aug.json") as f:
        dev_data = json.load(f)
    tgt_dictionary = Dictionary.load("./data/ja.dict")
    src_dictionary = Dictionary.load("./data/en.dict")
    train_loader = MatchLoader(train_data, src_dictionary, tgt_dictionary, args.cuda, True, args.max_cand)
    dev_loader = MatchLoader(dev_data, src_dictionary, tgt_dictionary, args.cuda, True, args.max_cand)

    # create model
    model = BinaryModel(len(src_dictionary), args.word_dim, len(tgt_dictionary), args.word_dim, args.hidden_dim, args.dropout)
    if args.cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    report_freq = args.report_freq
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    best_record = 0
    for epoch in range(args.epoch):
        avg_loss = 0
        step_cnt = 0
        # train
        model.train()
        for (i, batch) in enumerate(train_loader):
            if batch["sents"].size(1) == 0:
                continue
            pred_scores = model(batch["src"], batch["sents"])
            loss = loss_fn(pred_scores, batch["scores"])
            avg_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            step_cnt += 1
            if step_cnt == report_freq:
                print("epoch: %d, batch: %d, loss: %.6f"%(epoch, i + 1, avg_loss / report_freq))
                step_cnt = 0
                avg_loss = 0
        # evaluate
        model.eval()
        with torch.no_grad():
            avg_loss = 0
            avg_acc = 0
            for (i, batch) in enumerate(dev_loader):
                pred_scores = model(batch["src"], batch["sents"])
                loss = loss_fn(pred_scores, batch["scores"])
                avg_loss += loss.item()
                pred_scores = torch.sigmoid(pred_scores)
                pred_scores = pred_scores.squeeze().cpu().numpy()
                real_scores = batch["scores"].squeeze().cpu().numpy()
                hit_num = ((pred_scores > 0.5) == real_scores).sum()
                avg_acc += hit_num / batch["sents"].size(0)
            avg_acc /= i
            print("dev loss: %.6f"%(avg_loss / i))
            print("dev acc: %.6f"%(avg_acc))
        scheduler.step(1 - avg_acc)
        if best_record < avg_acc:
            best_record = avg_acc
            print("best record: %.6f"%best_record)
            torch.save(model.state_dict(), "./cache/model_binary.pt")
            torch.save(optimizer.state_dict(), "./cache/optimizer_binary.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Parameter')
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument("--max_cand", default=1000, type=int)
    parser.add_argument("--report_freq", default=100, type=int)
    parser.add_argument("--epoch", default=100, type=int)
    args = parser.parse_args()
    with torch.cuda.device(args.gpuid):
        main(args)
