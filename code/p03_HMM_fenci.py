import numpy as np
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
np.random.seed(1)

small = 50000
training = True
show_loss = True
N_epoch = 10
Batch_size = 256
show_epoch = 1
H = 128
em_dim = 100
lr = 1e-2
fold=5
net = 'HMM'
check_path = './Checkpoints/'
filepath = check_path + 'p03_Fenci_state_dict_' + net + ' .pkl'

words = {}
max_seq = 0

# label2idx = {'B': 0, 'I': 1, 'S': 2, 'BOS': 3, 'EOS': 4}
# idx2label = {0: 'B', 1: 'I', 2: 'S', 3: 'BOS', 4: 'EOS'}

Sentences = []
Sentences_tag = []

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
                mmp.append('S')
            else:
                mmp.append('B')
                for _ in range(1, len(word)):
                    mmp.append('I')
            for w in word:
                if w in words:
                    words[w] += 1
                else:
                    words[w] = 1
                tmp.append(w)
        Sentences.append(tmp)   # 存下以字单位的sentence
        Sentences_tag.append(mmp)  #存下每个sentence中每个字对应的BIS标签
        max_seq = max(max_seq, len(tmp))
        assert len(mmp)==len(tmp)
        assert len(tmp)> 0 # 判断是否存在空的sentence
        # print(tmp)
        if idx > small:
            break

print('len(words):', len(words))
print('max_seq len:', max_seq)

for idx, sentence in enumerate(Sentences):
    print('-' * 80)
    print(sentence)
    print(Sentences_tag[idx])
    if idx > 5:
        break

Train_Sentences, Train_Sentences_tag = [], []
Valid_Sentences, Valid_Sentences_tag = [], []
for i in range(len(Sentences)):
    if i % fold:
        Train_Sentences.append(Sentences[i])
        Train_Sentences_tag.append(Sentences_tag[i])
    else:
        Valid_Sentences.append(Sentences[i])
        Valid_Sentences_tag.append(Sentences_tag[i])

loss_fn = nn.CrossEntropyLoss(reduction='none')

def train(Train_Sentences, Train_Sentences_tag):
    N = len(Train_Sentences)
    tag, tag2word, tag2tag = {}, {}, {}
    tag['BOS'] = N
    tag['EOS'] = N
    for i in range(N):
        sentence = Train_Sentences[i]
        sentence_tag = Train_Sentences_tag[i]
        n = len(sentence)
        assert len(sentence) == len(sentence_tag)
        assert n > 0
        if ('BOS', sentence_tag[0]) in tag2tag:
            tag2tag[('BOS', sentence_tag[0])] += 1
        else:
            tag2tag[('BOS', sentence_tag[0])] = 1
        if (sentence_tag[-1], 'EOS') in tag2tag:
            tag2tag[(sentence_tag[-1], 'EOS')] += 1
        else:
            tag2tag[(sentence_tag[-1], 'EOS')] = 1

        for i in range(n):
            tg, w = sentence_tag[i], sentence[i]
            if tg in tag:
                tag[tg] += 1
            else:
                tag[tg] = 1
            if (tg, w) in tag2word:
                tag2word[(tg, w)] += 1
            else:
                tag2word[(tg, w)] = 1

            if i < n - 1:
                next_tg = sentence_tag[i + 1]
                if (tg, next_tg) in tag2tag:
                    tag2tag[(tg, next_tg)] += 1
                else:
                    tag2tag[(tg, next_tg)] = 1
    Prob_tag2tag, Prob_tag2word = {}, {}
    for tg1, tg2 in tag2tag.keys():
        Prob_tag2tag[(tg1, tg2)] = 0.0 + tag2tag[(tg1, tg2)] / tag[tg1]
    for tg, w in tag2word.keys():
        Prob_tag2word[(tg, w)] = 0.0 + tag2word[(tg, w)] / tag[tg]
    # print('tag:{} \ntag2word:{} \ntag2tag:{} \n'.format(tag, tag2word, tag2tag))
    print('tag:{} \ntag2word:{} \ntag2tag:{} \n'.format(len(tag), len(tag2word), len(tag2tag)))
    # print('\nProb_tag2word:{} \nProb_tag2tag:{} \n'.format(Prob_tag2word, Prob_tag2tag))
    print('\nProb_tag2word:{} \nProb_tag2tag:{} \n'.format(len(Prob_tag2word), len(Prob_tag2tag)))
    return tag, tag2word, tag2tag, Prob_tag2tag, Prob_tag2word
Tag, Tag2word, Tag2tag, Prob_tag2tag, Prob_tag2word = train(Train_Sentences, Train_Sentences_tag)

def predict_tag(sentence, True_sentence_tag=None):
    n = len(sentence)
    tags = ['B', 'I', 'S', 'BOS', 'EOS']
    dp = [{'B': 0.0, 'I': 0.0, 'S': 0.0, 'BOS': 0.0, 'EOS': 0.0} for _ in range(n + 1)]
    pre_tag = [{'B': None, 'I': None, 'S': None, 'BOS': None, 'EOS': None} for _ in range(n + 1)]
    for t in range(n):
        w = sentence[t]
        # print('w:', w)
        for tg in tags:
            prob_tag2word = 1e-9 if (tg, w) not in Prob_tag2word else Prob_tag2word[(tg, w)]
            if t == 0:
                prob_tag2tag = 1e-9 if ('BOS', tg) not in Prob_tag2tag else Prob_tag2tag[('BOS', tg)]
                dp[t][tg] = np.log(prob_tag2tag) + np.log(prob_tag2word)
                pre_tag[t][tg] = 'BOS'
            else:
                max_prob = None
                best_pre_tag = None
                for pre_tg in tags:
                    prob_tag2tag = 1e-9 if (pre_tg, tg) not in Prob_tag2tag else Prob_tag2tag[(pre_tg, tg)]
                    tmp = dp[t - 1][pre_tg] + np.log(prob_tag2tag) + np.log(prob_tag2word)
                    if max_prob == None or max_prob < tmp:
                        max_prob = tmp
                        best_pre_tag = pre_tg
                dp[t][tg] = max_prob
                pre_tag[t][tg] = best_pre_tag

    max_prob = None
    best_pre_tag = None
    tg = 'EOS'
    for pre_tg in tags:
        prob_tag2tag = 1e-9 if (pre_tg, tg) not in Prob_tag2tag else Prob_tag2tag[(pre_tg, tg)]
        tmp = dp[n - 1][pre_tg] + np.log(prob_tag2tag)
        if max_prob == None or max_prob < tmp:
            max_prob = tmp
            best_pre_tag = pre_tg
    dp[n][tg] = max_prob
    pre_tag[n][tg] = best_pre_tag

    ans_tag = []
    t = n

    # print('#' * 80)
    # print('sentence:', sentence)
    # print('True sentence tag:', True_sentence_tag)
    # print('len(sentence):', len(sentence))
    # print('n:', n)
    if True_sentence_tag is not None:
        True_sentence_tag.append('EOS')
    sss = sentence + ['END']
    while pre_tag[t][tg] is not None:
        if True_sentence_tag is None:
            # print('t: {}, pre_tag[t][tg]: {} -> tg: {} -- word:{}'.format(
            #     t,  pre_tag[t][tg], tg, sss[t]))
            pass
        else:
            assert len(True_sentence_tag) == n + 1, (n, len(True_sentence_tag))
            print('t: {}, pre_tag[t][tg]: {} -> tg: {}  -- True tag: {}, -- word: {}'.format(
                t, pre_tag[t][tg], tg, True_sentence_tag[t], sss[t]))

        ans_tag = [pre_tag[t][tg]] + ans_tag
        tg = pre_tag[t][tg]
        t = t - 1

    return ans_tag[1:]  # 去掉BOS

# predict_tag(sentence=Sentences[0], True_sentence_tag=Sentences_tag[0])
predict_tag(sentence=Sentences[0], True_sentence_tag=None)


def fenci_example():

    test_sentences = [
        '我是中国人，我爱祖国',
        '独行侠队的球员们承诺每天为达拉斯地区奋战在抗疫一线的工作人员们提供餐食',
        '汤普森太爱打球,不能出场让他很煎熬',
        '这个赛季对克莱来说非常艰难，他太热爱打篮球了，无法上场让他很受打击。',
        '克莱和斯蒂芬会处在极佳的状态，准备好比赛。',
        '勇士已经证明了他们也是一支历史级别的球队，维金斯在稍强于巴恩斯的前提下，仍然算得上是三号位上一位合格的替代者'
    ]

    np.set_printoptions(precision=3, suppress=True)

    for sentence in test_sentences:
        print('-' * 80)
        print('test word : {}'.format(sentence))
        sentence = [w for w in sentence]
        sentence_tag = predict_tag(sentence)
        # predict_tag(sentence=Sentences[0], True_sentence_tag=None)
        for i, w in enumerate(sentence):
            print('{} -> {}'.format(w, sentence_tag[i]))
fenci_example()

print('end!!!')
'''
test word : 克莱和斯蒂芬会处在极佳的状态，准备好比赛。
克 -> B
莱 -> I
和 -> S
斯 -> B
蒂 -> I
芬 -> I
会 -> S
处 -> B
在 -> I
极 -> B
佳 -> I
的 -> S
状 -> B
态 -> I
， -> S
准 -> B
备 -> I
好 -> S
比 -> B
赛 -> I
。 -> S
'''