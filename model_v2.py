import torch
#color prediction
import torch
import time
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.init as init
import torch.nn as nn
import csv
from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Linear, ReLU, Softmax, Module, CrossEntropyLoss
from torch.nn.init import kaiming_uniform_, xavier_uniform_#用于初始化神经网络模型中的权重参数


def lossMME(y_true, y_pred):

    m = y_true.shape[0]
    mu = torch.mean(y_true)

    mape = torch.mean(torch.abs(y_true - y_pred) / y_true)
    mae = torch.mean(torch.abs(y_true - y_pred))
    sd = torch.std(y_true - y_pred)

    alpha = 3
    beta = 2

    loss = alpha * mape + beta * sd + 0.000015 * torch.norm(y_pred, p=1) + 0.000003 * torch.norm(y_pred, p=2)

    return loss



# 定义模型
class SWPMPredictionModel(nn.Module):
    def __init__(self):
        super(SWPMPredictionModel, self).__init__()
        self.fc1 = nn.Linear(207, 100)
        self.fc2 = nn.Linear(100, 90)
        self.fc3 = nn.Linear(90, 90)
        self.fc4 = nn.Linear(90, 80)
        self.fc5 = nn.Linear(80, 80)
        self.fc6 = nn.Linear(80, 70)
        self.fc7 = nn.Linear(70, 70)
        self.fc8 = nn.Linear(70, 60)
        self.fc9 = nn.Linear(60, 60)
        self.fc10 = nn.Linear(60, 60)
        self.fc11 = nn.Linear(60, 60)
        self.fc12 = nn.Linear(60, 50)
        self.fc13 = nn.Linear(50, 50)
        self.fc14 = nn.Linear(50, 50)
        self.fc15 = nn.Linear(50, 50)
        self.fc16 = nn.Linear(50, 41)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        x = self.sigmoid(self.fc6(x))
        x = self.sigmoid(self.fc7(x))
        x = self.sigmoid(self.fc8(x))
        x = self.sigmoid(self.fc9(x))
        x = self.sigmoid(self.fc10(x))
        x = self.sigmoid(self.fc11(x))
        x = self.sigmoid(self.fc12(x))
        x = self.sigmoid(self.fc13(x))
        x = self.sigmoid(self.fc14(x))
        x = self.sigmoid(self.fc15(x))
        x = self.fc16(x)
        return x

# 定义损失函数和优化器
model = SWPMPredictionModel()
model.load_state_dict(torch.load('model_epoch_50000.pth'))
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
path1 = './data/MQ_train_1956_X_ok.csv'
path2= './data/MQ_train_1956_Y_ok.csv'
path='C:/Users/hydro1/lowen/color/data/trainXY.csv'
# 读取训练数据和标签数据
train_data = pd.read_csv(path1)
label_data = pd.read_csv(path2)

# 将训练数据和标签数据转换为 NumPy 数组
train_data_array = train_data.values
label_data_array = label_data.values

# 将 NumPy 数组转换为 PyTorch 张量
train_data_tensor = torch.tensor(train_data_array, dtype=torch.float32)
label_data_tensor = torch.tensor(label_data_array, dtype=torch.float32)

# 将训练数据和标签数据划分为批次
batch_size = 128
train_data_batches = torch.utils.data.DataLoader(train_data_tensor, batch_size=batch_size)
label_data_batches = torch.utils.data.DataLoader(label_data_tensor, batch_size=batch_size)

# 训练模型

for epoch in range(3700000):
    for batch_idx, (inputs, labels) in enumerate(zip(train_data_batches, label_data_batches)):
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = lossMME(outputs, labels)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            current_time = time.strftime('%m-%d__%H:%M:%S')
            print('Epoch {}, Loss: {} , Time: {}'.format(epoch, loss.item(),current_time))

        if epoch % 10000 == 0:
            torch.save(model.state_dict(), 'model_epoch_{}.pth'.format(epoch))