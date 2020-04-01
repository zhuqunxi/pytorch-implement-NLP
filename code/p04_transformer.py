import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print('device:', device)
device = torch.device(device)
cfg={}
cfg['device'] = device
'''
1. Word Embedding + position Embeding + mask
2. Encoder layer 
    Multihead  - num_head 个 attention 
    add + norm
    feed-forward MN
'''

def position_embed(max_seq_len, d_model):
    #  sin(pos / 10000^(2i/d_model))
    #  cos(pos / 10000^(2i/d_model))
    def get_angle(pos):
        return pos / 10000 ** (2 * np.arange(d_model // 2) / d_model)
    Ang = []
    for pos in range(max_seq_len):
        Ang.append(get_angle(pos))
    cos = np.cos(Ang)
    sin = np.sin(Ang)
    pos_emd = np.concatenate([cos, sin], axis=1)
    # plt.pcolormesh(pos_emd)
    # plt.show()
    print('pos_emd shape:', pos_emd.shape)
    return torch.Tensor(pos_emd)

class Multi_attention(nn.Module):
    def __init__(self, num_head, d_model):
        super(Multi_attention, self).__init__()
        self.Wq = nn.Linear(in_features=d_model, out_features=d_model)
        self.Wk = nn.Linear(in_features=d_model, out_features=d_model)
        self.Wv = nn.Linear(in_features=d_model, out_features=d_model)
        self.Wo = nn.Linear(in_features=d_model, out_features=d_model)
        self.num_head = num_head
        self.d_head = d_model // num_head
        self.d_model = d_model

        assert d_model % num_head == 0

    def split_head(self, X):
        batch = X.size(0)
        #  X: [batch, seq_len, d_model]
        X = X.view(batch, -1, self.num_head, self.d_head)  # [batch, seq_len, num_head,  d_model]
        X = X.transpose(2, 1)
        return X #  X: [batch, num_head, seq_len, d_head]

    def scale_att(self, Q, K, V, mask=None):
        # Q:[batch, num_head, seq_len_q, d_head]
        # K:[batch, num_head, seq_len_k, d_head]
        # V:[batch, num_head, seq_len_v, d_head],  seq_len_k = seq_len_v
        # mask: [batch, 1, seq_len_v, seq_len_k]
        d_q = Q.size(-1)
        K = K.transpose(3, 2)
        # [batch, num_head, seq_len_q, seq_len_k]

        # print('Q:{}\nK:{}\nd_q:{}'.format(Q.shape, K.shape, d_q))
        QK_scale_dot = torch.matmul(Q, K)
        QK_scale_dot = QK_scale_dot / torch.sqrt(torch.Tensor([d_q])).to(cfg['device'])
        if mask is not None:
            QK_scale_dot += mask * 1e-9
        att_W = F.softmax(QK_scale_dot, dim=-1)

        # [batch, num_head, seq_len_q, d_head]
        out = torch.matmul(att_W, V)
        return out, att_W

    def forward(self, Q, K, V, mask=None):
        # Q : [batch, seq_len_q, d_q]
        # K : [batch, seq_len_k, d_k]
        # V : [batch, seq_len_v, d_v]
        batch = Q.size(0)
        Q = self.Wq(Q)
        K = self.Wq(K)
        V = self.Wq(V)

        # [batch, num_head, seq_len, d_head]
        Q = self.split_head(Q)
        K = self.split_head(K)
        V = self.split_head(V)

        # out:[batch, num_head, seq_len_k, d_head]
        # att_W :[batch, num_head, seq_len_q, seq_len_k]
        out, att_W = self.scale_att(Q, K, V, mask=mask)
        out = out.transpose(2, 1)  # [batch, seq_len_q, num_head, d_head ]
        out = out.reshape(batch, -1, self.d_model)
        return out, att_W

class FFN(nn.Module):
    def __init__(self, d_model, d_ffn):
        super(FFN, self).__init__()
        self.dense1 = nn.Linear(in_features=d_model, out_features=d_ffn)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(in_features=d_ffn, out_features=d_model)
    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))

class Laynorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(Laynorm, self).__init__()
        self.a = nn.Parameter(torch.rand(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self, x):
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        x_std = torch.std(x, dim=-1, keepdim=True)
        # print('x_mean:', x_mean.size())
        # print('x_std:', x_std.size())
        # print('x:', x.size())
        # print('self.a:', self.a.size())
        # print('self.b:',self.b.size())
        return self.a * ((x - x_mean) / (x_std + self.eps)) + self.b

class Encoder_layer(nn.Module):
    def __init__(self, d_model, num_head, d_ffn, drop=0.1):
        super(Encoder_layer, self).__init__()
        self.mha = Multi_attention(num_head=num_head, d_model=d_model)
        self.laynorm1 = Laynorm(d_model=d_model)
        self.laynorm2 = Laynorm(d_model=d_model)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        self.ffn = FFN(d_model=d_model, d_ffn=d_ffn)
    def forward(self, x, mask=None):
        att_out, att_W = self.mha(Q=x, K=x, V=x, mask=mask)
        att_out = self.drop1(att_out)
        x = self.laynorm1(x + att_out)
        ffn_out = self.ffn(x)
        ffn_out = self.drop2(ffn_out)
        x = self.laynorm2(x + ffn_out)
        return x

class Encoder(nn.Module):
    def __init__(self, num_ec_layer, d_model, num_head, d_ffn, drop=0.1):
        super(Encoder, self).__init__()
        self.Ecoder_layers = nn.ModuleList(Encoder_layer(d_model=d_model, num_head=num_head, d_ffn=d_ffn, drop=drop)
                                           for _ in range(num_ec_layer))
        self.num_ec_layers = num_ec_layer
    def forward(self, x, mask=None):
        for i in range(self.num_ec_layers):
            x = self.Ecoder_layers[i](x, mask=mask)
        return x

class Transformer_EC(nn.Module):
    def __init__(self, max_seq_len, num_ec_layer, d_model, num_head, d_ffn, num_tag, word_size, emd_size):
        super(Transformer_EC, self).__init__()
        self.embed = nn.Embedding(num_embeddings=word_size, embedding_dim=emd_size)

        # [1, max_seq_len, d_model]
        self.pos_emd = position_embed(max_seq_len=max_seq_len, d_model=d_model).unsqueeze(0)

        self.ec_model = Encoder(num_ec_layer=num_ec_layer, d_model=d_model, num_head=num_head, d_ffn=d_ffn)
        self.dense = nn.Linear(in_features=d_model, out_features=num_tag)
    def forward(self, x, mask=None):
        # x: [batch, seq_len, 1]
        x = self.embed(x)   #x: [batch, seq_len, d_model]
        pos = self.pos_emd[:, :x.size(1), :].to(cfg['device'])
        # print('x shape:', x.size())
        # print('pos shape:', pos.shape)
        x = x + pos
        x = self.ec_model(x,mask=mask)
        x = self.dense(x)
        return x


if __name__=='__main__':
    position_embed(max_seq_len=500, d_model=512)

    batch, seq_len = 1, 2
    num_head, d_model = 2, 4
    d_ffn = 8
    num_ec_layer = 2
    max_seq_len = 5
    num_tag = 3
    word_size = 10
    emd_size = d_model
    Q, K, V = torch.rand([batch, seq_len, d_model]), torch.rand([batch, seq_len, d_model]), torch.rand([batch, seq_len, d_model])
    mha = Multi_attention(num_head=num_head, d_model=d_model)
    out, att_W = mha(Q, K, V)
    print('out, att_W shape:', out.size(), att_W.size())

    model = Encoder_layer(d_model=d_model, num_head=num_head, d_ffn=d_ffn)
    out = model(x=Q)
    print('after 1 Ecoder layer, out shape:', out.size())


    model = Encoder(num_ec_layer=num_ec_layer, d_model=d_model, num_head=num_head, d_ffn=d_ffn)
    out = model(x=Q)
    print('after Ecoder, out shape:', out.size())

    Q = torch.randint(0, 10, size=[batch, seq_len])
    model = Transformer_EC(max_seq_len, num_ec_layer, d_model, num_head, d_ffn, num_tag, word_size, emd_size)
    out = model(x=Q)
    print('after model_fc, out shape:', out.size())
    pass