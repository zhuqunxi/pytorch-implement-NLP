import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import time
import matplotlib.pyplot as plt

np.random.seed(1)
torch.manual_seed(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print('device:', device)
device = torch.device(device)

small = 50000
training = True
show_loss = True
N_epoch = 10
Batch_size = 256
show_epoch = 1
H = 128
em_dim = 100
lr = 1e-2
fold=0
net = 'BiLSTM'


words = {}
max_seq = 0

label2idx = {'B': 0, 'I': 1, 'S': 2}
idx2label = {0: 'B', 1: 'I', 2: 'S'}

Sentences = []
Sentences_label = []
Sentences_origin = []

with open('zhihu.txt', mode='r', encoding='utf8') as f:
    lines = f.readlines()
    print('len(lines):', len(lines))
    for idx, line in enumerate(lines):
        # print(line)
        line = line.split()
        tmp = []
        mmp = []
        for word in line:
            if len(word)==1:
                mmp.append(label2idx['S'])
            else:
                mmp.append(label2idx['B'])
                for _ in range(1, len(word)):
                    mmp.append(label2idx['I'])

            for w in word:
                if w in words:
                    words[w] += 1
                else:
                    words[w] = 1
                tmp.append(w)
        Sentences.append(tmp)
        Sentences_label.append(mmp)
        max_seq = max(max_seq, len(tmp))
        assert len(mmp)==len(tmp)
        # print(tmp)
        if idx > small:
            break

Sentences.sort(key=lambda x: len(x), reverse=True)
Sentences_label.sort(key=lambda x: len(x), reverse=True)
# for sentence in Sentences:
#     print(sentence)
print('len(words):', len(words))
# print(words)
print('max_seq len:', max_seq)

# print(words)
word2index = {word: idx + 1 for idx, word in enumerate(words.keys())}
indx2word = {idx + 1: word for idx, word in enumerate(words.keys())}


voc_size = len(words) + 1
word2index['<pad>'] = 0
indx2word[0] = '<pad>'

Sentences_idx, Sentences_len = [], []
for sentence in Sentences:
    tmp=[]
    for w in sentence:
        tmp.append(word2index[w])
    Sentences_idx.append(torch.LongTensor(tmp))
    Sentences_len.append(len(tmp))
    # print(tmp)
Sentences_idx = torch.nn.utils.rnn.pad_sequence(Sentences_idx,batch_first=True)
# print('-' * 80)
# print(Sentences_idx.size())
# print(Sentences_idx)


# print('-' * 80)
Sentences_label_idx = []
for i, sentences_label in enumerate(Sentences_label):
    tmp = torch.LongTensor(sentences_label)
    Sentences_label_idx.append(tmp)
    # print(Sentences[i])
    # # print(lines[i])
    # print(tmp)
    assert len(tmp) == len(Sentences[i])
Sentences_label_idx = torch.nn.utils.rnn.pad_sequence(Sentences_label_idx,batch_first=True,padding_value=0)
# print('Sentences_label_idx:')
# print(Sentences_label_idx)

# a = torch.tensor(1.0)
# print(a)
# print(a.size())
class MyDataSet(Dataset):
    def __init__(self, data, lens, labels):
        self.data = data
        self.lens = lens
        self.labels = labels
    def __getitem__(self, idx):
        now_data = self.data[idx]
        now_len = self.lens[idx]
        now_mask = []
        now_label = self.labels[idx]
        for i in range(len(now_data)):
            t = 1.0 if i < now_len else 0.0
            now_mask.append(t)
        now_mask = torch.Tensor(now_mask)
        return now_data, now_len, now_mask, now_label
    def __len__(self):
        return len(self.data)

class FenCi_Zqx(nn.Module):
    def __init__(self, voc_size, em_dim, H):
        super(FenCi_Zqx, self).__init__()
        self.emd = nn.Embedding(num_embeddings=voc_size,embedding_dim=em_dim)
        if net == 'LSTM':
            self.rnn = nn.LSTM(input_size=em_dim,hidden_size=H,num_layers=1,batch_first=True)
            self.linear = nn.Linear(in_features=H,out_features=3)
        if net == 'BiLSTM':
            self.rnn = nn.LSTM(input_size=em_dim, hidden_size=H, num_layers=1, batch_first=True,bidirectional=True)
            self.linear = nn.Linear(in_features=2*H, out_features=3)
    def forward(self, sentence, sentence_len=None, mask=None):
        emd = self.emd(sentence) #  (batch, seq_len, em_dim)
        all_h, (h, c) = self.rnn(emd) # LSTM: (batch, seq_len, H)   BiLSTM: (batch, seq_len, 2*H)
        # print('emd size:', emd.size())
        # print('all_h.size():', all_h.size())
        # out = all_h.view(-1, all_h.size(2))  # (batch * seq_len, H)
        out = self.linear(all_h).view(emd.size(0), emd.size(1), 3) # (batch, seq_len, 3)
        # print('out size:', out.size())
        return out

Sentences_len = torch.Tensor(Sentences_len)
train_idx = [i for i in range(len(Sentences_len)) if i % 5 == fold]
test_idx = [i for i in range(len(Sentences_len)) if i % 5 != fold]
print(train_idx, '\n', test_idx)
Train_Sentences_idx, Train_Sentences_len, Train_Sentences_label_idx = \
    Sentences_idx[train_idx], Sentences_len[train_idx], Sentences_label_idx[train_idx]
Test_Sentences_idx, Test_Sentences_len, Test_Sentences_label_idx = \
    Sentences_idx[test_idx], Sentences_len[test_idx], Sentences_label_idx[test_idx]
Train_data = MyDataSet(Train_Sentences_idx, Train_Sentences_len, Train_Sentences_label_idx)
Train_data_loader = DataLoader(dataset=Train_data, batch_size=Batch_size, shuffle=True)
Test_data = MyDataSet(Test_Sentences_idx, Test_Sentences_len, Test_Sentences_label_idx)
Test_data_loader = DataLoader(dataset=Test_data, batch_size=Batch_size, shuffle=False)

model = FenCi_Zqx(voc_size=voc_size, em_dim=em_dim, H=H)
loss_fn = nn.CrossEntropyLoss(reduction='none')
opt = torch.optim.Adam(model.parameters(), lr=lr)


print('Sentences_idx, Sentences_len, Sentences_label_idx shape')
print(len(Sentences_idx), len(Sentences_len), len(Sentences_label_idx))
print(Sentences_idx.size(), Sentences_len.size(), Sentences_label_idx.size())
print(Sentences_idx.shape, Sentences_len.shape, Sentences_label_idx.shape)
print('#' * 60)
print(model)

def valid(model):
    # model.to(device)
    # model.eval()
    with torch.no_grad():
        avg_loss = 0
        cnt=0
        for batch_step, (now_data, now_len, now_mask, now_label) in enumerate(Test_data_loader):
            cnt += 1
            now_data, now_len, now_mask, now_label = \
                now_data.to(device), now_len.to(device), now_mask.to(device), now_label.to(device)
            out = model(now_data, now_len, now_mask)
            out = out.view(-1, 3)
            now_mask = now_mask.view(-1)
            now_label = now_label.view(-1)
            loss = loss_fn(out, now_label)
            # print('loss size:', loss.size())
            # print(out.size(), now_label.size(), now_mask.size())
            loss = torch.mean(loss * now_mask)
            avg_loss += loss.item()
            # print('loss size:', loss.size())
            # print('loss:', loss.item())

        avg_loss /= cnt
        return avg_loss


def train(model):
    print('start training:')
    model.to(device)
    time_st_global = time.time()
    Train_loss,Valid_loss = [], []
    for epoch in range(N_epoch):
        time_st_epoch = time.time()
        avg_loss = 0
        cnt = 0
        for batch_step, (now_data, now_len, now_mask, now_label) in enumerate(Train_data_loader):
            cnt += 1
            now_data, now_len, now_mask, now_label = \
                now_data.to(device), now_len.to(device), now_mask.to(device), now_label.to(device)
            out = model(now_data, now_len, now_mask)
            out = out.view(-1, 3)
            now_mask = now_mask.view(-1)
            now_label = now_label.view(-1)
            loss = loss_fn(out, now_label)
            # print('loss size:', loss.size())
            # print(out.size(), now_label.size(), now_mask.size())
            loss = torch.mean(loss * now_mask)
            avg_loss += loss.item()
            # print('loss size:', loss.size())
            # print('loss:', loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()
        avg_loss /= cnt
        valid_avg_loss = valid(model)
        print('#' * 80)
        print('epoch:{}, steps: {}, train avg loss: {} -- valid avg loss : {} '.format(epoch, cnt, avg_loss, valid_avg_loss))
        print('epoch time {:.2f} s, total time {:.2f} s'.format(time.time() - time_st_epoch,
                                                                time.time() - time_st_global))
        if len(Valid_loss)==0 or valid_avg_loss < min(Valid_loss):
            if not os.path.exists(check_path):
                os.makedirs(check_path)
            torch.save(model.state_dict(), filepath)

        Train_loss.append(avg_loss)
        Valid_loss.append(valid_avg_loss)

    if show_loss:
        plt.figure()
        plt.plot(Train_loss,label='Train loss')
        plt.plot(Valid_loss, label='Valid loss')
        plt.legend()
        plt.savefig('Train_Valid_loss' + net + '.png')
        # plt.show()
    return model

        # break

check_path = './Checkpoints/'
filepath = check_path + 'p03_Fenci_state_dict_' + net + ' .pkl'

if training:
    model = train(model)


# 模型恢复
model.load_state_dict(torch.load(filepath))


test_words = [
    '我是中国人，我爱祖国',
    '独行侠队的球员们承诺每天为达拉斯地区奋战在抗疫一线的工作人员们提供餐食',
    '汤普森太爱打球,不能出场让他很煎熬',
    '这个赛季对克莱来说非常艰难，他太热爱打篮球了，无法上场让他很受打击。',
    '克莱和斯蒂芬会处在极佳的状态，准备好比赛。',
    '勇士已经证明了他们也是一支历史级别的球队，维金斯在稍强于巴恩斯的前提下，仍然算得上是三号位上一位合格的替代者'
]

np.set_printoptions(precision=3, suppress=True)
model.cpu()
model.eval()
for word in test_words:
    print('-' * 80)
    print('test word : {}'.format(word))
    word_idx = [word2index[w] for w in word]
    word_idx = torch.LongTensor([word_idx])
    # print('word_idx.size():', word_idx.size())
    # word_idx.to(device)
    out = model(word_idx)
    # print('out.size():', out.size())
    out = out.squeeze(0).cpu().detach().numpy()
    # print('out.shape():', out.shape)
    # print(out)
    out_label = np.argmax(out, axis=1)
    # print(out_label)

    for i, w in enumerate(word):
        print('{} -> {} -> {}'.format(w, idx2label[out_label[i]], out_label[i]))

print('end!!!')
'''
test word : 勇士已经证明了他们也是一支历史级别的球队，维金斯在稍强于巴恩斯的前提下，仍然算得上是三号位上一位合格的替代者
勇 -> B -> 0
士 -> I -> 1
已 -> B -> 0
经 -> I -> 1
证 -> B -> 0
明 -> I -> 1
了 -> S -> 2
他 -> S -> 2
们 -> I -> 1
也 -> S -> 2
是 -> S -> 2
一 -> S -> 2
支 -> S -> 2
历 -> B -> 0
史 -> I -> 1
级 -> B -> 0
别 -> I -> 1
的 -> S -> 2
球 -> B -> 0
队 -> I -> 1
， -> S -> 2
维 -> B -> 0
金 -> I -> 1
斯 -> I -> 1
在 -> S -> 2
稍 -> B -> 0
强 -> B -> 0
于 -> I -> 1
巴 -> B -> 0
恩 -> I -> 1
斯 -> I -> 1
的 -> S -> 2
前 -> B -> 0
提 -> I -> 1
下 -> S -> 2
， -> S -> 2
仍 -> B -> 0
然 -> I -> 1
算 -> I -> 1
得 -> I -> 1
上 -> S -> 2
是 -> S -> 2
三 -> S -> 2
号 -> S -> 2
位 -> S -> 2
上 -> B -> 0
一 -> B -> 0
位 -> S -> 2
合 -> B -> 0
格 -> I -> 1
的 -> S -> 2
替 -> B -> 0
代 -> I -> 1
者 -> I -> 1

'''