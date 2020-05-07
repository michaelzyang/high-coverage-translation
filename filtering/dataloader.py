import random
import torch

def pad_seq(X, pad=0):
    """
    zero-pad sequnces to same length then stack them together
    """  
    maxlen = max([x.size(0) for x in X])
    Y = []
    for x in X:
        padlen = maxlen - x.size(0)
        if padlen > 0:
            paddings = torch.ones(padlen, requires_grad=True).type(x.type()) * pad
            x_ = torch.cat((x, paddings), 0)
            Y.append(x_)
        else:
            Y.append(x)
    return torch.stack(Y)
    

class MatchLoader(object):
    """
    base loader
    """
    def __init__(self, data, src_dictionary, tgt_dictionary, use_gpu=False, shuffle=False, max_cand=100):
        """
        data format: [[{"id", "tokens", "score"}...]...] 
        """
        self.data = data
        self.src_dictionary = src_dictionary
        self.tgt_dictionary = tgt_dictionary
        self.use_gpu = use_gpu
        self.shuffle = shuffle
        self.max_cand = max_cand
        self.count = 0
        self.len = len(self.data)
        if self.shuffle:
            random.shuffle(self.data)
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.count == self.len:
            self.count = 0
            if self.shuffle:
                random.shuffle(self.data)
            raise StopIteration()
        else:
            batch = self.data[self.count]
            src = [self.src_dictionary.word2idx[w] for w in batch["src"]]
            src = torch.LongTensor(src).unsqueeze(0)
            batch = batch["cand"]
            # random.shuffle(batch)
            batch = batch[:self.max_cand]
            sents = [[self.tgt_dictionary.word2idx[w] for w in x["tokens"]] for x in batch]
            sents = pad_seq([torch.LongTensor(x) for x in sents], self.tgt_dictionary.pad_token_id)
            scores = [1 if x["score"] > 0 else 0 for x in batch]
            scores = torch.FloatTensor(scores)
            ids = [x["id"] for x in batch]
            if self.use_gpu:
                sents = sents.cuda()
                scores = scores.cuda()
                src = src.cuda()
            self.count += 1
            return {"sents": sents, "scores": scores, "ids": ids, "src": src}

    def __len__(self):
        return self.len

    def get(self):
        try:
            data = self.__next__()
        except StopIteration:
            data = self.__next__()
        return data