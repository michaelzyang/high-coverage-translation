import pickle

class Dictionary():
    def __init__(self, fpath=None):
        with open(fpath) as f:
            lines = f.readlines()
        self.word2idx = {x.split("\t")[0]:i+1 for (i, x) in enumerate(lines)}
        self.word2idx["<pad>"] = 0
        self.idx2word = {v:k for (k, v) in self.word2idx.items()}
        self.len = len(list(self.idx2word.keys()))

    @property
    def eos_token(self):
        return "<\s>"

    @property
    def eos_token_id(self):
        return self.word2idx[self.eos_token]

    @property
    def bos_token(self):
        return "<s>"

    @property
    def bos_token_id(self):
        return self.word2idx[self.bos_token]

    @property
    def pad_token(self):
        return "<pad>"

    @property
    def pad_token_id(self):
        return self.word2idx[self.pad_token]

    @property
    def unk_token(self):
        return "<unk>"

    @property
    def unk_token_id(self):
        return self.word2idx[self.unk_token]

    @classmethod
    def load(cls, fpath):
        with open(fpath, "rb") as f:
            model = pickle.load(f)
        return model

    def __len__(self):
        return self.len

    def save(self, fpath):
        with open(fpath, "wb") as f:
            pickle.dump(self, f)

if __name__ == "__main__":
    model = Dictionary('data/courses/en-ja/bin/enja_spm_models/spm.en.nopretok.vocab')
    model.save("./data/en.dict")