from keras.datasets import mnist
import torch
import numpy as np
np.random.seed(1)
torch.manual_seed(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print('device:', device)
device = torch.device(device)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('x_train, y_train shape:', x_train.shape, y_train.shape)
print('x_test, y_test shape:', x_test.shape, y_test.shape)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print('x_train, y_train shape:', x_train.shape, y_train.shape)
print('x_test, y_test shape:', x_test.shape, y_test.shape)

N, D_in, D_out = 1000, x_train.shape[1], 10

class Two_layer(torch.nn.Module):
    def __init__(self, D_in, D_out, H=1000):
        super(Two_layer, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(H, D_out)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = Two_layer(D_in, D_out, H = 1000)
loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=model.parameters(), lr=1e-4)
x_train, y_train = torch.tensor(x_train,dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
x_test, y_test = torch.tensor(x_test,dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)
model = model.to(device)
import time
time_st = time.time()
for epoch in range(50):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)

    if not epoch % 10:
        with torch.no_grad():
            y_pred_test = model(x_test)
            y_label_pred = np.argmax(y_pred_test.cpu().detach().numpy(), axis=1)
            print('y_label_pred y_test shape:', y_label_pred.shape, y_test.size())
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

print('training time used {:.2f} s with device {}'.format(time.time() - time_st, device))

'''
x_train, y_train shape: (60000, 28, 28) (60000,)
x_test, y_test shape: (10000, 28, 28) (10000,)
x_train, y_train shape: (60000, 784) (60000,)
x_test, y_test shape: (10000, 784) (10000,)
y_label_pred y_test shape: (10000,) torch.Size([10000])
test loss: 23.847854614257812, acc: 0.1414
train loss: 23.87252426147461, acc: 0.13683333333333333
--------------------------------------------------------------------------------
y_label_pred y_test shape: (10000,) torch.Size([10000])
test loss: 3.340665578842163, acc: 0.7039
train loss: 3.514056444168091, acc: 0.6925166666666667
--------------------------------------------------------------------------------
y_label_pred y_test shape: (10000,) torch.Size([10000])
test loss: 1.7213207483291626, acc: 0.844
train loss: 1.8277908563613892, acc: 0.84025
--------------------------------------------------------------------------------
y_label_pred y_test shape: (10000,) torch.Size([10000])
test loss: 1.2859240770339966, acc: 0.8845
train loss: 1.3402273654937744, acc: 0.88125
--------------------------------------------------------------------------------
y_label_pred y_test shape: (10000,) torch.Size([10000])
test loss: 1.0803418159484863, acc: 0.8993
train loss: 1.084514856338501, acc: 0.8984833333333333
--------------------------------------------------------------------------------
training time used 81.26 s with device cpu
training time used 3.61 s with device cuda
'''