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
show_acc = True
reused_W = False
N_epoch = 1
Batch_size = 256
show_epoch = 10
H = 128
em_dim = 100
lr = 1e-2
fold=5
net = 'HMM'
check_path = './Checkpoints/'
filepath = check_path + 'W_crf.npy'
regulization = 0

words = {}
max_seq = 0

# label2idx = {'B': 0, 'I': 1, 'S': 2, 'BOS': 3, 'EOS': 4}
# idx2label = {0: 'B', 1: 'I', 2: 'S', 3: 'BOS', 4: 'EOS'}
tag_num = 5
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

def predict_tag(W, feat_pair2idx, sentence, True_sentence_tag=None):
    n = len(sentence)
    tags = ['B', 'I', 'S', 'BOS', 'EOS']
    dp = [{'B': 0.0, 'I': 0.0, 'S': 0.0, 'BOS': 0.0, 'EOS': 0.0} for _ in range(n + 1)]
    pre_tag = [{'B': None, 'I': None, 'S': None, 'BOS': None, 'EOS': None} for _ in range(n + 1)]
    for t in range(n):
        w = sentence[t]
        # print('w:', w)
        for tg in tags:
            feat_tag2word = -1e9 if (tg, w) not in feat_pair2idx else W[feat_pair2idx[(tg, w)]]
            if t == 0:
                feat_tag2tag = -1e9 if ('BOS', tg) not in feat_pair2idx else W[feat_pair2idx[('BOS', tg)]]
                dp[t][tg] = feat_tag2word + feat_tag2tag
                pre_tag[t][tg] = 'BOS'
            else:
                max_prob = None
                best_pre_tag = None
                for pre_tg in tags:
                    feat_tag2tag = -1e9 if (pre_tg, tg) not in feat_pair2idx else W[feat_pair2idx[(pre_tg, tg)]]
                    tmp = dp[t - 1][pre_tg] + feat_tag2tag + feat_tag2word
                    if max_prob == None or max_prob < tmp:
                        max_prob = tmp
                        best_pre_tag = pre_tg
                dp[t][tg] = max_prob
                pre_tag[t][tg] = best_pre_tag

    max_prob = None
    best_pre_tag = None
    tg = 'EOS'
    for pre_tg in tags:
        feat_tag2tag = -1e9 if (pre_tg, tg) not in feat_pair2idx else W[feat_pair2idx[(pre_tg, tg)]]
        tmp = dp[n - 1][pre_tg] + feat_tag2tag
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


def cal_grad_w(W, feat_pair2idx, feat_num, xn, yn):
    """
    O(W, xn, yn) = log p(yn|xn) = log exp(W Phi(xn, yn)) / \Sigma exp(W Phi(xn, y'))
        =W Phi(xn, yn) - log \Sigma exp(W Phi(xn, y'))
        = W Phi(xn, yn) - log Z(xn)
    W_grad = Phi(xn, yn) - 1 / Z(xn)  * \Sigma exp(W Phi(xn, y')) Phi(xn, y')
    然后利用viterbi算法进行求解，实际上就是 O( 序列长度 * tag种类数 ** 2)的动态规划算法
    我们用到两个东西：
        1. Z_i(t)：表示t时刻为止，tag是i的所有路径的概率之和，
                i.e.,  \Sigma exp(W Phi(xn(1:t), y'(1:t))) 且y(t) = tag i
        2. P_i(t): 表示t时刻为止，tag是i的所有路径的【加权】（Phi(xn(1:t), y'(1:t))）概率之和，
                i.e.,  \Sigma exp(W Phi(xn(1:t), y'(1:t))) Phi(xn(1:t), y'(1:t))
    具体状态转移方程见代码， 关键是
        P_i(t + 1) = exp(W Phi(xn(1:t+1), y'(1:t+1))) Phi(xn(1:t+1), y'(1:t+1))
            = exp(W Phi_t) exp (W delta_Phi) * (Phi_t +delta_Phi)
            =  \Sigma_{y'(t)}  (exp(W Phi_t)Phi_t + delta_Phi) * exp (W delta_Phi)
        Z_i(t + 1) = \Sigma_{y'(t)} Z_i(t) * exp (W delta_Phi)
    为了数值稳定，可以用log_P和log_Z进行更新
    如果看不懂上面，可以参考下面的链接（可能还是比较模糊），最好自己推导一边
    链接1：https://blog.csdn.net/qq_42189083/article/details/89350890
    链接2：https://blog.csdn.net/weixin_30014549/article/details/52850638
    """
    tags = ['B', 'I', 'S', 'BOS', 'EOS']
    Phi = np.zeros(feat_num)
    pre_P = np.zeros(shape=[5, feat_num])
    pre_Z = np.zeros(shape=[5,])
    n = len(xn)

    pre_tag = 'BOS'
    for i in range(n):
        word, tag = xn[i], yn[i]
        tag2tag_id = feat_pair2idx[(pre_tag, tag)]
        tag2word_id = feat_pair2idx[(tag, word)]
        Phi[tag2tag_id] += 1
        Phi[tag2word_id] += 1
        pre_tag = tag

    for i in range(n):
        word = xn[i]

        P = np.zeros(shape=[5, feat_num])
        Z = np.zeros(shape=[5, ])
        flag = 0
        for j, tag in enumerate(tags):
            for k, pre_tag in enumerate(tags):
                if i==0 and pre_tag != 'BOS':
                    continue
                deta_phi = np.zeros(feat_num)
                tag2tag = (pre_tag, tag)
                tag2word = (tag, word)
                if tag2tag not in feat_pair2idx:
                    continue
                if tag2word not in feat_pair2idx:
                    continue
                flag = 1
                tag2tag_id = feat_pair2idx[tag2tag]
                tag2word_id = feat_pair2idx[tag2word]
                deta_phi[tag2tag_id] += 1
                deta_phi[tag2word_id] += 1

                # exp_w_delta_phi = np.exp(np.sum(W * deta_phi))
                exp_w_delta_phi = np.exp(W[tag2tag_id] + W[tag2word_id])

                if i == 0 and pre_tag == 'BOS':
                    pre_Z[k] = 1
                P[j] += (pre_P[k] + pre_Z[k] * deta_phi) * exp_w_delta_phi
                Z[j] += pre_Z[k] * exp_w_delta_phi

                # print('P[j, tag2tag_id]:{}, P[j, tag2word_id]:{}'.format(P[j, tag2tag_id], P[j, tag2word_id]))
        pre_P = P.copy()
        pre_Z = Z.copy()
        # print('word: {}, flag: {}'.format(word, flag))

    P = np.zeros(shape=[feat_num, ])
    Z = 0.0
    tag = 'EOS'
    for k, pre_tag in enumerate(tags):
        deta_phi = np.zeros(feat_num)
        tag2tag = (pre_tag, tag)
        if tag2tag not in feat_pair2idx:
            continue
        tag2tag_id = feat_pair2idx[tag2tag]
        deta_phi[tag2tag_id] += 1
        # exp_w_delta_phi = np.exp(np.sum(W * deta_phi))
        exp_w_delta_phi = np.exp(W[tag2tag_id])

        P += (pre_P[k] + pre_Z[k] * deta_phi) * exp_w_delta_phi
        Z += pre_Z[k] * exp_w_delta_phi
    # print('pre_P: {}\npre_Z: {}\n'.format(pre_P, pre_Z))
    # print('sum(Phi): {}\nP:{}\nZ:{}'.format(np.sum(Phi), P, Z))
    # print('WPhi: {}, exp(WPhi):{}'.format(np.sum(W * Phi), np.exp(np.sum(W * Phi))))
    # print('Phi - P / Z:', Phi - P / Z)
    W_grad = Phi - P / Z
    return - W_grad + regulization * W

def cal_grad_w_log_version(W, feat_pair2idx, feat_num, xn, yn):
    """
    O(W, xn, yn) = log p(yn|xn) = log exp(W Phi(xn, yn)) / \Sigma exp(W Phi(xn, y'))
        =W Phi(xn, yn) - log \Sigma exp(W Phi(xn, y'))
        = W Phi(xn, yn) - log Z(xn)
    W_grad = Phi(xn, yn) - 1 / Z(xn)  * \Sigma exp(W Phi(xn, y')) Phi(xn, y')
    然后利用viterbi算法进行求解，实际上就是 O( 序列长度 * tag种类数 ** 2)的动态规划算法
    我们用到两个东西：
        1. Z_i(t)：表示t时刻为止，tag是i的所有路径的概率之和，
                i.e.,  \Sigma exp(W Phi(xn(1:t), y'(1:t))) 且y(t) = tag i
        2. P_i(t): 表示t时刻为止，tag是i的所有路径的【加权】（Phi(xn(1:t), y'(1:t))）概率之和，
                i.e.,  \Sigma exp(W Phi(xn(1:t), y'(1:t))) Phi(xn(1:t), y'(1:t))
    具体状态转移方程见代码， 关键是
        P_i(t + 1) = exp(W Phi(xn(1:t+1), y'(1:t+1))) Phi(xn(1:t+1), y'(1:t+1))
            = exp(W Phi_t) exp (W delta_Phi) * (Phi_t +delta_Phi)
            =  \Sigma_{y'(t)}  (exp(W Phi_t)Phi_t + delta_Phi) * exp (W delta_Phi)
        Z_i(t + 1) = \Sigma_{y'(t)} Z_i(t) * exp (W delta_Phi)
    为了数值稳定，可以用log_P和log_Z进行更新
    如果看不懂上面，可以参考下面的链接（可能还是比较模糊），最好自己推导一边
    链接1：https://blog.csdn.net/qq_42189083/article/details/89350890
    链接2：https://blog.csdn.net/weixin_30014549/article/details/52850638
    """
    tags = ['B', 'I', 'S', 'BOS', 'EOS']
    Phi = np.zeros(feat_num)
    log_pre_P = np.zeros(shape=[5, feat_num])
    log_pre_Z = np.zeros(shape=[5,])
    n = len(xn)

    pre_tag = 'BOS'
    for i in range(n):
        word, tag = xn[i], yn[i]
        tag2tag_id = feat_pair2idx[(pre_tag, tag)]
        tag2word_id = feat_pair2idx[(tag, word)]
        Phi[tag2tag_id] += 1
        Phi[tag2word_id] += 1
        pre_tag = tag

    for i in range(n):
        word = xn[i]

        log_P = np.zeros(shape=[5, feat_num]) + 1e-9
        log_Z = np.zeros(shape=[5, ]) + 1e-9
        flag = 0
        for j, tag in enumerate(tags):

            for k, pre_tag in enumerate(tags):
                if i==0 and pre_tag != 'BOS':
                    continue
                deta_phi = np.zeros(feat_num)
                tag2tag = (pre_tag, tag)
                tag2word = (tag, word)
                if tag2tag not in feat_pair2idx:
                    continue
                if tag2word not in feat_pair2idx:
                    continue
                flag = 1
                tag2tag_id = feat_pair2idx[tag2tag]
                tag2word_id = feat_pair2idx[tag2word]
                deta_phi[tag2tag_id] += 1
                deta_phi[tag2word_id] += 1

                # exp_w_delta_phi = np.exp(np.sum(W * deta_phi))
                exp_w_delta_phi = np.exp(W[tag2tag_id] + W[tag2word_id])

                if i == 0 and pre_tag == 'BOS':
                    log_pre_Z[k] = 0
                log_P[j] += (np.exp(log_pre_P[k]) + np.exp(log_pre_Z[k]) * deta_phi) * exp_w_delta_phi
                log_Z[j] += np.exp(log_pre_Z[k]) * exp_w_delta_phi

                # print('P[j, tag2tag_id]:{}, P[j, tag2word_id]:{}'.format(log_P[j, tag2tag_id], log_P[j, tag2word_id]))
        log_P = np.log(log_P)
        log_Z = np.log(log_Z)
        log_pre_P = log_P.copy()
        log_pre_Z = log_Z.copy()
        # print('word: {}, flag: {}'.format(word, flag))

    log_P = np.zeros(shape=[feat_num, ])
    log_Z = 0.0
    tag = 'EOS'
    for k, pre_tag in enumerate(tags):
        deta_phi = np.zeros(feat_num)
        tag2tag = (pre_tag, tag)
        if tag2tag not in feat_pair2idx:
            continue
        tag2tag_id = feat_pair2idx[tag2tag]
        deta_phi[tag2tag_id] += 1
        # exp_w_delta_phi = np.exp(np.sum(W * deta_phi))
        exp_w_delta_phi = np.exp(W[tag2tag_id])

        log_P += (np.exp(log_pre_P[k]) + np.exp(log_pre_Z[k]) * deta_phi) * exp_w_delta_phi
        log_Z += np.exp(log_pre_Z[k]) * exp_w_delta_phi
    # print('pre_P: {}\npre_Z: {}\n'.format(pre_P, pre_Z))
    # print('sum(Phi): {}\nP:{}\nZ:{}'.format(np.sum(Phi), P, Z))
    # print('WPhi: {}, exp(WPhi):{}'.format(np.sum(W * Phi), np.exp(np.sum(W * Phi))))
    # print('Phi - P / Z:', Phi - P / Z)
    W_grad = Phi - log_P / log_Z
    return - W_grad + regulization * W

def evaluate(W, feat_pair2idx, Sentences_, Sentences_tag_):
    cnt_correct_tag, cnt_total_tag = 0.0, 0.0
    for i, sentence in enumerate(Sentences_):
        sentence_tag = Sentences_tag_[i]
        sentence_tag_pred = predict_tag(W, feat_pair2idx, sentence)
        assert len(sentence_tag) == len(sentence_tag_pred)
        # predict_tag(sentence=Sentences[0], True_sentence_tag=None)
        # print('sentence_tag == sentence_tag_pred:', [sentence_tag[_] == sentence_tag_pred[_] for _ in range(len(sentence))])
        cnt_correct_tag += np.sum([sentence_tag[_] == sentence_tag_pred[_] for _ in range(len(sentence))])
        cnt_total_tag += len(sentence)
        # for j, w in enumerate(sentence):
        #     print('w:{} -> true_tag:{} -> pred_tag:{}'.format(w, sentence_tag[j], sentence_tag_pred[j]))
        # break
    acc = cnt_correct_tag / cnt_total_tag
    # print('cnt_correct_tag, cnt_total_tag:', cnt_correct_tag, cnt_total_tag)
    # print('acc:', acc)
    return acc

def train(Train_Sentences, Train_Sentences_tag):
    '''
    :param Train_Sentences:
    :param Train_Sentences_tag:
    p(Sentences_tag, Sentences) ~  exp(w^T f(Sentences_tag, Sentences)), w是待train的权重，f是特征函数
    :return:
    '''
    N = len(Train_Sentences)
    def get_feature_dict():
        feat_pair2idx = {}
        feat_idx2pair = {}
        feat_num = 0
        for i in range(N):
            sentence = Train_Sentences[i]
            sentence_tag = Train_Sentences_tag[i]
            n = len(sentence)
            pre_tg = 'BOS'
            for i in range(n):
                tg, w = sentence_tag[i], sentence[i]
                if (tg, w) not in feat_pair2idx:
                    feat_pair2idx[(tg, w)] = feat_num
                    feat_idx2pair[feat_num] = (tg, w)
                    feat_num += 1
                if (pre_tg, tg) not in feat_pair2idx:
                    feat_pair2idx[(pre_tg, tg)] = feat_num
                    feat_idx2pair[feat_num] = (pre_tg, tg)
                    feat_num += 1
                pre_tg = tg
            tg = 'EOS'
            if (pre_tg, tg) not in feat_pair2idx:
                feat_pair2idx[(pre_tg, tg)] = feat_num
                feat_idx2pair[feat_num] = (pre_tg, tg)
                feat_num += 1
        return feat_pair2idx, feat_idx2pair, feat_num

    feat_pair2idx, feat_idx2pair, feat_num = get_feature_dict()
    print('{}\n{}\n{}\n'.format(feat_pair2idx, feat_idx2pair, feat_num))

    if reused_W:
        W = np.load(filepath)
    else:
        W = np.random.normal(0, scale=1.0 / np.sqrt(feat_num), size=[feat_num, ])
    # tag, tag2word, tag2tag = {}, {}, {}
    # tag['BOS'] = N
    # tag['EOS'] = N
    Train_Acc, Valid_Acc = [], []
    time_global = time.time()
    for epoch in range(N_epoch):
        time_epoch = time.time()
        s = '###'

        for i in range(N):
            if i % (N // 10)==0:
                s_out = s * (i // (N // 10)) + '{}/{} running this epoch time used: {:.2f}'.format(i, N, time.time() - time_epoch)
                if i // (N // 10) == 10:
                    print(s_out, end="", flush=False)
                else:
                    print(s_out, end="\r", flush=True)
            sentence = Train_Sentences[i]
            sentence_tag = Train_Sentences_tag[i]
            n = len(sentence)
            assert len(sentence) == len(sentence_tag)
            assert n > 0
            W_grad = cal_grad_w(W, feat_pair2idx, feat_num, xn=sentence, yn=sentence_tag)
            # W_grad = cal_grad_w_log_version(W, feat_pair2idx, feat_num, xn=sentence, yn=sentence_tag)

            W -= lr * W_grad
        train_acc = evaluate(W, feat_pair2idx, Sentences_=Train_Sentences, Sentences_tag_=Train_Sentences_tag)
        valid_acc = evaluate(W, feat_pair2idx, Sentences_=Valid_Sentences, Sentences_tag_=Valid_Sentences_tag)
        Train_Acc.append(train_acc)
        Valid_Acc.append(valid_acc)
        print('\nepoch: {}, epoch time: {}, global time: {}, train acc: {}, valid acc: {}'.format(
            epoch, time.time() - time_epoch, time.time() - time_global, train_acc, valid_acc))

    if show_acc:
        plt.figure()
        plt.title('regulization: {}'.format(regulization))
        plt.plot(Train_Acc, label='Train Acc')
        plt.plot(Valid_Acc, label='Valid Acc')



    return W, feat_pair2idx

REG = [0, 0.1, 0.3, 1, 3, 10, 30]
for reg in REG:
    regulization = reg
    W, feat_pair2idx = train(Train_Sentences, Train_Sentences_tag)
    break
plt.show()
if not os.path.exists(check_path):
    os.makedirs(check_path)
np.save(filepath, W)

# predict_tag(W, feat_pair2idx, sentence=Sentences[0], True_sentence_tag=None)

predict_tag(W, feat_pair2idx, sentence=Sentences[0], True_sentence_tag=Sentences_tag[0])
# predict_tag(sentence=Sentences[0], True_sentence_tag=None)


def fenci_example(W, feat_pair2idx):

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
        sentence_tag = predict_tag(W, feat_pair2idx, sentence)
        # predict_tag(sentence=Sentences[0], True_sentence_tag=None)
        for i, w in enumerate(sentence):
            print('{} -> {}'.format(w, sentence_tag[i]))

fenci_example(W, feat_pair2idx)

print('end!!!')
'''
test word : 独行侠队的球员们承诺每天为达拉斯地区奋战在抗疫一线的工作人员们提供餐食
独 -> B
行 -> I
侠 -> B
队 -> I
的 -> S
球 -> B
员 -> I
们 -> I
承 -> B
诺 -> I
每 -> B
天 -> I
为 -> B
达 -> I
拉 -> B
斯 -> I
地 -> B
区 -> I
奋 -> B
战 -> I
在 -> S
抗 -> B
疫 -> I
一 -> B
线 -> I
的 -> S
工 -> B
作 -> I
人 -> S
员 -> B
们 -> I
提 -> B
供 -> I
餐 -> B
食 -> I
'''