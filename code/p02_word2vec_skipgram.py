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

small = 5000000
training = False
N_epoch = 51
Batch_size = 128
show_epoch = 1

words = {}
with open('zhihu.txt', mode='r', encoding='utf8') as f:
    lines = f.readlines()
    print('len(lines):', len(lines))
    for idx, line in enumerate(lines):
        # print(line)
        for word in line.split():
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
        if idx > small:
            break
print(len(words))
# print(words)
word2index = {word: idx for idx, word in enumerate(words.keys())}
indx2word = {idx: word for idx, word in enumerate(words.keys())}
# print(word2index)
# print(indx2word)
word_freq = np.array(list(words.values()))
word_freq = word_freq / np.sum(word_freq)
word_freq = word_freq ** (3 / 4.0)
word_freq = word_freq / np.sum(word_freq)
word_freq = torch.Tensor(word_freq)
# print(word_freq)

C, K = 3, 10 # C:窗口大小， K:每个positive样本对应K个negative样本
em_dim = 100
word_size = len(words)


def creat_train_data():
    Center_Outside_words, Center_Outside_words_index = [], []
    with open('zhihu.txt', mode='r', encoding='utf8') as f:
        lines = f.readlines()
        print('len(lines):', len(lines))
        for _, line in enumerate(lines):
            # print(line)
            line = line.split()
            n = len(line)
            for idx, word in enumerate(line):
                st = max(idx - C, 0)
                ed = min(idx + 1 + C, n)
                for i in range(st, idx):
                    word_ = line[i]
                    Center_Outside_words.append([word, word_])
                    Center_Outside_words_index.append([word2index[word], word2index[word_]])
                for i in range(idx + 1, ed):
                    word_ = line[i]
                    Center_Outside_words.append([word, word_])
                    Center_Outside_words_index.append([word2index[word], word2index[word_]])
            if _ > small:
                break
    return Center_Outside_words, Center_Outside_words_index

Center_Outside_words, Center_Outside_words_index = creat_train_data()
Center_Outside_words_index = np.array(Center_Outside_words_index)

print(Center_Outside_words[:10])
print(Center_Outside_words_index[:10])
print('train data len:', len(Center_Outside_words))

N_train = len(Center_Outside_words)

def get_batch(batch_step):
    st, ed = batch_step * Batch_size, min(batch_step * Batch_size + Batch_size, N_train)
    assert st < ed
    center_word = torch.LongTensor(Center_Outside_words_index[st:ed, 0])  # (batch, )
    outside_word = torch.LongTensor(Center_Outside_words_index[st:ed, 1]) # (batch, )
    negtive_word = torch.multinomial(word_freq, K * (ed - st)).view(-1, K) # (batch, K)

    # print(center_word.size(), outside_word.size(), negtive_word.size())
    # print(center_word, outside_word, negtive_word)
    return center_word, outside_word, negtive_word

center_word, outside_word, negtive_word = get_batch(batch_step=0)

class Zhihu_DataSet(Dataset):
    def __init__(self, Center_Outside_words_index, word_freq):
        self.Center_Outside_words_index = Center_Outside_words_index
        self.word_freq = word_freq
        print('Center_Outside_words_index shape:', Center_Outside_words_index.shape)

    def __len__(self):
        return len(self.Center_Outside_words_index)

    def __getitem__(self, index):
        # center_word = torch.LongTensor([self.Center_Outside_words_index[index, 0]])
        # outside_word = torch.LongTensor([self.Center_Outside_words_index[index, 1]])

        center_word = torch.tensor(self.Center_Outside_words_index[index, 0],dtype=torch.long)
        outside_word = torch.tensor(self.Center_Outside_words_index[index, 1],dtype=torch.long)

        negtive_word = torch.multinomial(word_freq, K, replacement=True)  # (batch, K)
        # print(center_word.size(), outside_word.size(), negtive_word.size())
        return center_word, outside_word, negtive_word



zhihu_dataset = Zhihu_DataSet(Center_Outside_words_index, word_freq)
zhihu_dataloader = DataLoader(dataset=zhihu_dataset,batch_size=Batch_size, shuffle=True)

class Word2Vec_Zqx(nn.Module):
    def __init__(self, word_size, em_dim):
        super(Word2Vec_Zqx, self).__init__()
        self.word_em_center = nn.Embedding(num_embeddings=word_size,embedding_dim=em_dim)
        self.word_em_outside = nn.Embedding(num_embeddings=word_size,embedding_dim=em_dim)

    def forward(self, center_word, outside_word, negtive_word):
        center_word_emd = self.word_em_center(center_word)  # (batch, em_dim)
        outside_word_emd = self.word_em_outside(outside_word) # (batch, em_dim)
        negtive_word_emd = self.word_em_outside(negtive_word) # (batch, K, em_dim))

        # print(center_word_emd.size(), outside_word_emd.size(), negtive_word_emd.size())
        center_word_emd = center_word_emd.unsqueeze(dim=2)  # (batch, em_dim, 1)
        outside_word_emd = outside_word_emd.unsqueeze(dim=1)  # (batch, 1, em_dim)
        # print(center_word_emd.size(), outside_word_emd.size(), negtive_word_emd.size())
        center_outside_word = torch.bmm(outside_word_emd, center_word_emd).squeeze(1)
        center_outside_word = center_outside_word.squeeze(1)  # (batch, )
        center_negtive_word = torch.bmm(negtive_word_emd, center_word_emd).squeeze(2)  # (batch, K)
        # print(center_outside_word.size(), center_negtive_word.size())

        loss = - (torch.sum(F.logsigmoid(center_outside_word)) + torch.sum(F.logsigmoid(center_negtive_word)))
        return loss

    def get_emd_center(self):
        return self.word_em_center.weight.cpu().detach().numpy()

model =Word2Vec_Zqx(word_size=word_size, em_dim=em_dim)
loss = model(center_word, outside_word, negtive_word)
print('loss:', loss.item())

# 模型保存
check_path = './Checkpoints/'
filepath = check_path + 'word2vec_state_dict.pkl'
def find_similar_word(emd_center, word):
    word_idx = word2index[word]
    word_emd = emd_center[word_idx].reshape(-1, 1)
    # similarity = np.matmul(emd_center, word_emd).flatten()
    similarity = np.matmul(emd_center, word_emd).flatten() / np.linalg.norm(emd_center, axis=1) / np.linalg.norm(word_emd)
    k = 10
    topk_idx = np.argsort(-similarity)[:k]

    print('与word=[{}]--相似的top {}的有:'.format(word, k))
    topk_word = [indx2word[_] for _ in topk_idx]
    print(topk_word)

def train(model):
    # opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    model.to(device)
    import time
    time_st_global = time.time()
    for epoch in range(N_epoch):
        time_st_epoch = time.time()
        for batch_step in range(N_train // Batch_size):
            center_word, outside_word, negtive_word = get_batch(batch_step)
            center_word, outside_word, negtive_word = center_word.to(device), outside_word.to(device), negtive_word.to(device)
            loss = model(center_word, outside_word, negtive_word)

            opt.zero_grad()
            loss.backward()
            opt.step()
        print('# ' * 80)
        print('epoch:{}, batch_step: {}, loss: {}'.format(epoch, batch_step, loss.item()))
        print('epoch time {:.2f} s, total time {:.2f} s'.format(time.time() - time_st_epoch, time.time() - time_st_global))
        if not epoch % show_epoch:
            if not os.path.exists(check_path):
                os.makedirs(check_path)
            torch.save(model.state_dict(), filepath)
            emd_center = model.get_emd_center()

            test_words = ['你', '为什么', '学生', '女生', '什么', '大学']
            for word in test_words:
                print('-' * 80)
                print('test word : {},  次数: {}'.format(word, words[word]))
                find_similar_word(emd_center=emd_center, word=word)

    return model

def train_with_dataloader(model):
    # opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    model.to(device)
    import time
    time_st_global = time.time()
    for epoch in range(N_epoch):
        time_st_epoch = time.time()
        for batch_step, (center_word, outside_word, negtive_word) in enumerate(zhihu_dataloader):
            center_word, outside_word, negtive_word = center_word.to(device), outside_word.to(device), negtive_word.to(device)
            loss = model(center_word, outside_word, negtive_word)

            opt.zero_grad()
            loss.backward()
            opt.step()
        print('#' * 80)
        print('epoch:{}, batch_step: {}, loss: {}'.format(epoch, batch_step, loss.item()))
        print('epoch time {:.2f} s, total time {:.2f} s'.format(time.time() - time_st_epoch, time.time() - time_st_global))
        if not epoch % show_epoch:
            if not os.path.exists(check_path):
                os.makedirs(check_path)
            torch.save(model.state_dict(), filepath)

            # emd_center = model.get_emd_center()
            # test_words = ['你', '为什么', '学生', '女生', '什么', '大学']
            # for word in test_words:
            #     print('-' * 80)
            #     print('test word : {},  次数: {}'.format(word, words[word]))
            #     find_similar_word(emd_center=emd_center, word=word)

    return model

if training:
    # model = train(model)
    model = train_with_dataloader(model)

# 模型恢复
model.load_state_dict(torch.load(filepath))
loss = model(center_word, outside_word, negtive_word)
print('loss:', loss.item())
emd_center = model.get_emd_center()


test_words = ['你', '为什么', '学生', '女生', '什么', '大学']

for word in test_words:
    print('-' * 80)
    print('test word : {},  次数: {}'.format(word, words[word]))
    find_similar_word(emd_center=emd_center, word=word)

print('end!!!')