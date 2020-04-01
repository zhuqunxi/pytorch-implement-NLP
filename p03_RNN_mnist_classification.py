from keras.datasets import mnist
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
np.random.seed(1)
torch.manual_seed(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print('device:', device)
device = torch.device(device)

N, D_in, D_out = 10000, 28, 10
H = 100
Batch_size = 128
lr=1e-2
N_epoch = 200


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = x_train[:N], y_train[:N]
x_test, y_test = x_test[:N], y_test[:N]

# 归一化很重要，不然有可能train不起来，或者test效果不行
x_train = x_train /255.0
x_test = x_test / 255.0

print('x_train, y_train shape:', x_train.shape, y_train.shape)
print('x_test, y_test shape:', x_test.shape, y_test.shape)
print('np.max(x_train), np.min(x_train):', np.max(x_train), np.min(x_train))
print('np.max(y_train), np.min(y_train):', np.max(y_train), np.min(y_train))

class RNN_zqx(nn.Module):
    def __init__(self, D_in, H):
        super(RNN_zqx, self).__init__()
        self.rnn = nn.LSTM(input_size=D_in,hidden_size=H,num_layers=1,batch_first=True)
        self.linear = nn.Linear(H, 10)
    def forward(self, x):
        all_h, (h, c) = self.rnn(x)
        # all_h: (batch, seq_len, num_directions * hidden_size)
        # h: (num_layers * num_directions, batch, hidden_size)
        # print('all_h.size():', all_h.size())
        # print('h.size():', h.size())
        x = self.linear(h.squeeze(0))
        return x

model =RNN_zqx(D_in=D_in, H=H)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=lr)

x_train, y_train = torch.Tensor(x_train), torch.LongTensor(y_train)
x_test, y_test = torch.Tensor(x_test), torch.LongTensor(y_test)

print('x_train.size(), y_train.size():', x_train.size(), y_train.size())
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)
mdoel = model.to(device)

time_st = time.time()
for epoch in range(N_epoch):
    y_pred = model(x_train)
    # print(y_pred.size())
    loss = loss_fn(y_pred, y_train)

    if not epoch % 10:
        with torch.no_grad():
            y_pred_test = model(x_test)
            y_label_pred = np.argmax(y_pred_test.cpu().detach().numpy(), axis=1)
            # print('y_label_pred y_test shape:', y_label_pred.shape, y_test.size())
            acc_test = np.mean(y_label_pred == y_test.cpu().detach().numpy())
            loss_test = loss_fn(y_pred_test, y_test)
            print('test loss: {}, acc: {}'.format(loss_test.item(), acc_test))

            y_label_pred_train = np.argmax(y_pred.cpu().detach().numpy(), axis=1)
            acc_train = np.mean(y_label_pred_train == y_train.cpu().detach().numpy())
            print('train loss: {}, acc: {}'.format(loss.item(), acc_train))

            print('-' * 80)

    opt.zero_grad()
    loss.backward()
    opt.step()

print('Training time used {:.2f} s'.format(time.time() - time_st))

'''
device: cuda
x_train, y_train shape: (10000, 28, 28) (10000,)
x_test, y_test shape: (10000, 28, 28) (10000,)
np.max(x_train), np.min(x_train): 1.0 0.0
np.max(y_train), np.min(y_train): 9 0
x_train.size(), y_train.size(): torch.Size([10000, 28, 28]) torch.Size([10000])
test loss: 2.3056862354278564, acc: 0.1032
train loss: 2.3057758808135986, acc: 0.0991
--------------------------------------------------------------------------------
test loss: 1.6542853116989136, acc: 0.5035
train loss: 1.651445746421814, acc: 0.482
--------------------------------------------------------------------------------
test loss: 1.0779469013214111, acc: 0.6027
train loss: 1.0364742279052734, acc: 0.6158
--------------------------------------------------------------------------------
test loss: 0.7418596148490906, acc: 0.7503
train loss: 0.7045448422431946, acc: 0.7642
--------------------------------------------------------------------------------
test loss: 0.5074136853218079, acc: 0.8369
train loss: 0.46816474199295044, acc: 0.8512
--------------------------------------------------------------------------------
test loss: 0.3507310748100281, acc: 0.8931
train loss: 0.29413318634033203, acc: 0.9125
--------------------------------------------------------------------------------
test loss: 0.25384169816970825, acc: 0.9292
train loss: 0.1905861645936966, acc: 0.9446
--------------------------------------------------------------------------------
test loss: 0.21215158700942993, acc: 0.9406
train loss: 0.13411203026771545, acc: 0.9614
--------------------------------------------------------------------------------
test loss: 0.19598548114299774, acc: 0.9467
train loss: 0.0968935638666153, acc: 0.9711
--------------------------------------------------------------------------------
test loss: 0.6670947074890137, acc: 0.834
train loss: 0.6392199993133545, acc: 0.8405
--------------------------------------------------------------------------------
test loss: 0.3550219237804413, acc: 0.8966
train loss: 0.29769250750541687, acc: 0.9112
--------------------------------------------------------------------------------
test loss: 0.22847041487693787, acc: 0.9345
train loss: 0.16787868738174438, acc: 0.9545
--------------------------------------------------------------------------------
test loss: 0.19370371103286743, acc: 0.9464
train loss: 0.1122715100646019, acc: 0.9692
--------------------------------------------------------------------------------
test loss: 0.16738709807395935, acc: 0.9538
train loss: 0.08012499660253525, acc: 0.9787
--------------------------------------------------------------------------------
test loss: 0.16035553812980652, acc: 0.9575
train loss: 0.06216369569301605, acc: 0.9838
--------------------------------------------------------------------------------
test loss: 0.15690605342388153, acc: 0.9587
train loss: 0.04842701926827431, acc: 0.9877
--------------------------------------------------------------------------------
test loss: 0.1597040444612503, acc: 0.9586
train loss: 0.03863723576068878, acc: 0.9909
--------------------------------------------------------------------------------
test loss: 0.16320295631885529, acc: 0.9593
train loss: 0.031261660158634186, acc: 0.9933
--------------------------------------------------------------------------------
test loss: 0.1675170212984085, acc: 0.959
train loss: 0.02533782459795475, acc: 0.9948
--------------------------------------------------------------------------------
test loss: 0.17022284865379333, acc: 0.9592
train loss: 0.020637042820453644, acc: 0.9962
--------------------------------------------------------------------------------
'''