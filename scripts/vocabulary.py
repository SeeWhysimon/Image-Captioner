import torch
import re
from collections import Counter
from typing import List

class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.min_freq = min_freq

    def build(self, captions):
        counter = Counter()
        for cap in captions:
            words = re.findall(r'\w+', cap.lower())
            counter.update(words)
        for word, freq in counter.items():
            if freq >= self.min_freq:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1

    def encode(self, caption):
        tokens = re.findall(r'\w+', caption.lower())
        ids = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in tokens]
        return torch.tensor([self.word2idx['<SOS>']] + ids + [self.word2idx['<EOS>']])
    
    def decode(self, tokens: List[int]) -> str:
        words = []
        for idx in tokens:
            if 0 <= idx < len(self.idx2word):
                word = self.idx2word[idx]
            else:
                word = "<UNK>"
            if word != "<PAD>":
                words.append(word)
        return " ".join(words)

    def __len__(self):
        return len(self.idx2word)
