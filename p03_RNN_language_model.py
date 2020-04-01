import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

np.random.seed(1)
torch.manual_seed(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print('device:', device)
device = torch.device(device)

small = 500
training = True
N_epoch = 5
Batch_size = 128
show_epoch = 1

words = {}
max_seq = 0
with open('zhihu.txt', mode='r', encoding='utf8') as f:
    lines = f.readlines()
    print('len(lines):', len(lines))
    for idx, line in enumerate(lines):
        # print(line)
        line = line.split()
        max_seq = max(max_seq, len(line))
        for word in line:
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
        if idx > small:
            break
print(len(words))
print('max_seq len:', max_seq)

# print(words)
word2index = {word: idx for idx, word in enumerate(words.keys())}
indx2word = {idx: word for idx, word in enumerate(words.keys())}

em_dim = 100
word_size = len(words) + 2
word2index['<start>'] = len(words)
word2index['<end>'] = len(words) + 1
indx2word[len(words)] = '<start>'
indx2word[len(words) + 1] = '<end>'

def creat_train_data():
    Sentences = []
    with open('zhihu.txt', mode='r', encoding='utf8') as f:
        lines = f.readlines()
        print('len(lines):', len(lines))
        for _, line in enumerate(lines):
            # print(line)
            line = line.split()
            n = len(line)
            
            if _ > small:
                break
    return Center_Outside_words, Center_Outside_words_index
