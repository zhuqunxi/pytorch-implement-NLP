import torch
import torch.nn as nn
import numpy as np
np.random.seed(1)
torch.manual_seed(1)

class CRF_zqx(nn.Module):
    def __init__(self, tag_num):
        super(CRF_zqx, self).__init__()
        # A为转移矩阵, A_ij, 表示tag i 到 tag j 的得分
        # self.A = torch.rand(size=(tag_num, tag_num), requires_grad=True)
        # self.A = nn.Parameter(torch.rand(size=(tag_num, tag_num)))
        self.A = nn.Parameter(torch.empty(tag_num, tag_num))
        self.tag_num = tag_num
        self.reset_parameters()
    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.A, -0.1, 0.1)

    def forward(self, y_pred, y_true, mask):
        if len(y_true.size()) < 3:
            # print(y_true.dtype)
            y_true = torch.nn.functional.one_hot(y_true, num_classes=self.tag_num)
            y_true = y_true.type(torch.float32)
        # y_pred, y_true: [batch_size, seq_len, tag_num],   ps:y_true是one-hot向量
        # log p(y_true | x_true) = log {exp(score(y_true, x_true) / \Sigma_y exp(score(y, x_true))}
        #                        = score(y_true, x_true) - log  sum_y exp(score(y, x_true))
        # print('forward:\n')
        # print('y_pred:{}\ny_true:{}\nmask:{}\n'.format(y_pred, y_true, mask))
        # print('y_pred:{}\ny_true:{}\nmask:{}\n'.format(y_pred.size(), y_true.size(), mask.size()))
        # print('A:', self.A)
        loss = self.score(y_pred, y_true, mask) - self.log_sum_exp(y_pred, mask)
        return torch.mean(-loss)

    def score(self, y_pred, y_true, mask):
        # y_pred, y_true: [batch_size, seq_len, tag_num]  mask: [batch_size, seq_len]
        mask = torch.unsqueeze(mask, dim=2)  #  mask: [batch_size, seq_len, 1]
        # print('y_pred, y_true, mask size:', y_pred.size(), y_true.size(), mask.size())
        score_word2tag = torch.sum(y_pred * y_true * mask, dim=[1, 2])  #  计算word2tag的分数,得到[batch_size, ]向量
        #            [batch_size, seq_len-1, tag_num, 1]  *  [batch_size, seq_len-1, 1, tag_num]
        #    从而获得[batch_size, seq_len-1, tag_num, tag_num], 后两个维度都是one-hot向量，分别表示tag2tag的转移矩阵A的index
        score_tag2tag = torch.unsqueeze(y_true[:, :-1, :] * mask[:, :-1, :], dim=3) \
                        * torch.unsqueeze(y_true[:, 1:, :] * mask[:, 1:, :], dim=2)

        #               [batch_size, seq_len-1, tag_num, tag_num]  *  [1, 1, tag_num, tag_num]
        A = torch.unsqueeze(torch.unsqueeze(self.A, 0), 0)
        score_tag2tag = score_tag2tag * A
        score_tag2tag = torch.sum(score_tag2tag, dim=[1, 2, 3])  # [batch_size,]
        score_ = score_word2tag + score_tag2tag
        # print('score_ size:', score_.size())
        # print('score:', score_)
        return score_

    def log_sum_exp(self, y_pred, mask):
        # mask: [batch_size, seq_len]
        seq_len = y_pred.size(1)
        pre_log_Z = y_pred[:, 0, :] # [batch_size, tag_num], initial: log Z = log exp(y_pred[time_step=0]) = y_pred[:, 0 , :]

        # print('pre_log_Z:{}, with size:{}'.format(pre_log_Z, pre_log_Z.size()))
        for i in range(1, seq_len):
            # print('i:', i)
            #                    [1, tag_num, tag_num]   +  [batch_size, tag_num, 1] = [batch_size, tag_num, tag_num]
            # 然后对列(dim=1)求logsumexp,  得到[batch_size, tag_num]
            tmp = pre_log_Z.unsqueeze(2)
            # log_Z = torch.logsumexp(tmp + self.A + y_pred[:, i:i+1, :], dim=1)
            log_Z = torch.logsumexp(torch.unsqueeze(self.A, 0) + torch.unsqueeze(pre_log_Z, 2), dim=1) + y_pred[:, i, :]
            log_Z = mask[:, i:i+1] * log_Z + (1 - mask[:, i:i+1]) * pre_log_Z  # 现在mask位置上是1，则更新， 如果是0，则取用pre_log_Z的值
            pre_log_Z = log_Z.clone()
        # print('log_Z size:', pre_log_Z.size())

        # print('res:', pre_log_Z)
        res = torch.logsumexp(pre_log_Z,dim=1)  # 是logsumexp  不是 sum,  debug了大半天！！！！
        # print('logsumexp:', res)
        return res

    def decode(self,y_pred, mask=None):
        batch, seq_len = y_pred.size(0), y_pred.size(1)
        if mask is None:
            mask = torch.ones(size=[batch, seq_len])

        pre_dp = y_pred[:, 0, :]  #[batch, tag_num]
        dp_best_idx = torch.LongTensor(torch.zeros(size=[batch, seq_len + 1, self.tag_num], dtype=torch.long) - 1)
        for i in range(1, seq_len):                      # from     to
            now_pred = y_pred[:, i:i+1, :]       # [batch, 1,       tag_num]
            pre_dp = torch.unsqueeze(pre_dp, 2)  # [batch, tag_num, 1      ]
            A = torch.unsqueeze(self.A, 0)       # [1,     tag_num, tag_num]
            dp, idx = torch.max(pre_dp + A + now_pred, dim=1) #  dp: [batch, tag_num]
            # print('dp:{}, idx:{}'.format(dp.size(), idx.size()))
            dp_best_idx[:, i, :] = idx
            pre_dp = dp.clone()

        best_value, last_tag = torch.max(pre_dp, dim=1)
        print('pre_dp:{}, pre_dp size:{}\npointer:{}, last_tag size:{}'.format(pre_dp, pre_dp.size(), last_tag, last_tag.size()))
        last_tag = list(last_tag.cpu().detach().numpy())
        dp_best_idx = dp_best_idx.cpu().detach().numpy()
        print('last tag:', last_tag)
        ans = [last_tag] # [batch]
        i = seq_len - 1
        while i:
            tmp = dp_best_idx[:, i, :]
            pre_tag = []
            for j in range(batch):
                pre_tag.append(tmp[j, last_tag[j]])
            last_tag = pre_tag.copy()
            ans = [pre_tag] + ans
            i -= 1
        ans = np.array(ans) #[seq_len, batch]
        ans = ans.transpose()
        print('ans:', ans)
        # while i:
        #     print('dp_best_idx[:, i, :] size:{}, pointer.unsqueeze(1) size:{}'.format(
        #         dp_best_idx[:, i, :].size(), pointer.unsqueeze(1).size()))
        #     print('dp_best_idx[:, i, :]:{}, pointer.unsqueeze(1):{}'.format(
        #         dp_best_idx[:, i, :], pointer.unsqueeze(1)))
        #     pointer = dp_best_idx[:, i, :][pointer.unsqueeze(1)]  # pointer.unsqueeze(1): [batch, 1]
        #     ans = [list(pointer)] + ans
        #     i = i - 1

        return ans

if __name__=='__main__':
    batch = 1
    seq_len = 3
    tag_num = 2
    y_pred = torch.rand(size=[batch, seq_len, tag_num])
    y_true = torch.randint(0, tag_num, size=[batch, seq_len])
    # print(y_true)
    y_true = torch.nn.functional.one_hot(y_true, num_classes=tag_num)
    y_true = y_true.type(torch.float32)
    # print(y_true)
    # print(y_true.size())
    mask = []
    for _ in range(batch):
        tmp = np.random.randint(2, seq_len)
        mask.append([1] * tmp + [0] * (seq_len - tmp))
    mask = torch.Tensor(mask)
    # print(mask)
    model =CRF_zqx(tag_num=tag_num)



    # print('y_pred:{}\ny_true:{}\nmask:{}\n'.format(y_pred, y_true, mask))
    # print(type(y_pred))
    # print(type(y_true))
    # print(type(mask))
    # print(y_pred.dtype)
    # print(y_true.dtype)
    # print(mask.dtype)

    print('y_pred=y_pred, y_true=y_true, mask=mask:', y_pred.size(), y_true.size(), mask.size())
    loss = model(y_pred=y_pred, y_true=y_true, mask=mask)
    print('loss: {}'.format(loss))


'''
y_pred:tensor([[[0.7576, 0.2793],
         [0.4031, 0.7347]]])
y_true:tensor([[[0., 1.],
         [0., 1.]]])
mask:tensor([[1., 0.]])

'''