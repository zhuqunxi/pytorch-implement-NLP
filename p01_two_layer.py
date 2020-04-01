import numpy as np
import torch
import torch.nn as nn

np.random.seed(1)
torch.manual_seed(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print('device:', device)
device = torch.device(device)

N, D_in, D_out =64, 1000, 10
train_x = np.random.normal(size=(N, D_in))
train_y = np.random.normal(size=(N, D_out))

class Two_layer(torch.nn.Module):
    def __init__(self, D_in, D_out, H=100):
        super(Two_layer, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(H, D_out)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = Two_layer(D_in, D_out, H=1000)

loss_fn = nn.MSELoss(reduction='sum')
# optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-4)

train_x = torch.from_numpy(train_x).type(dtype=torch.float32)
train_y = torch.from_numpy(train_y).type(dtype=torch.float32)

# train_x = torch.randn(N, D_in)
# train_y = torch.randn(N, D_out)

train_x = train_x.to(device)
train_y = train_y.to(device)
model = model.to(device)

import time
time_st = time.time()
for epoch in range(200):
    y_pred = model(train_x)

    loss = loss_fn(y_pred, train_y)

    optimizer.zero_grad()
    if not epoch % 20:
        # print('loss:', loss)
        print('loss.item(): ', loss.item())
    loss.backward()
    optimizer.step()
print('training time used {:.1f} s with {}'.format(time.time() - time_st, device))

"""
loss:  673.6837158203125
loss:  57.70276641845703
loss:  3.7402660846710205
loss:  0.2832883596420288
loss:  0.026732178404927254
loss:  0.0029198969714343548
loss:  0.00034921077894978225
loss:  4.434480797499418e-05
loss:  5.87546583119547e-06
loss:  8.037222301027214e-07
training time used 1.1 s with cpu
training time used 0.6 s with cuda
"""