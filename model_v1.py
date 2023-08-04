# -*- coding: utf-8 -*-
# pytorch mlp for multiclass classification
#color prediction
import torch
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


# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=0)
        # store the inputs and outputs
        self.X = df.values[:, :-41]
        self.y = df.values[:, -41:]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        # label encode target and ensure the values are floats
        # self.y = LabelEncoder().fit_transform(self.y)
        # self.y = LabelBinarizer().fit_transform(self.y)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.3):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 100)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(100, 90)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()

        self.hidden3 = Linear(90, 90)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()

        self.hidden4 = Linear(90, 80)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = ReLU()

        self.hidden5 = Linear(80, 80)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act5 = ReLU()

        self.hidden6 = Linear(80, 70)
        kaiming_uniform_(self.hidden6.weight, nonlinearity='relu')
        self.act6 = ReLU()

        self.hidden7 = Linear(70, 70)
        kaiming_uniform_(self.hidden7.weight, nonlinearity='relu')
        self.act7 = ReLU()

        self.hidden8 = Linear(70, 60)
        kaiming_uniform_(self.hidden8.weight, nonlinearity='relu')
        self.act8 = ReLU()

        self.hidden9 = Linear(60, 60)
        kaiming_uniform_(self.hidden9.weight, nonlinearity='relu')
        self.act9 = ReLU()

        self.hidden10 = Linear(60, 60)
        kaiming_uniform_(self.hidden10.weight, nonlinearity='relu')
        self.act10 = ReLU()

        self.hidden11 = Linear(60, 60)
        kaiming_uniform_(self.hidden11.weight, nonlinearity='relu')
        self.act11 = ReLU()

        self.hidden12 = Linear(60, 50)
        kaiming_uniform_(self.hidden12.weight, nonlinearity='relu')
        self.act12 = ReLU()

        self.hidden13 = Linear(50, 50)
        kaiming_uniform_(self.hidden13.weight, nonlinearity='relu')
        self.act13 = ReLU()

        self.hidden14 = Linear(50, 50)
        kaiming_uniform_(self.hidden14.weight, nonlinearity='relu')
        self.act14 = ReLU()

        self.hidden15 = Linear(50, 50)
        kaiming_uniform_(self.hidden15.weight, nonlinearity='relu')
        self.act15 = ReLU()
        # third hidden layer and output
        self.hidden16 = Linear(50, 41)# 41 R + 3 Lab = 44
        xavier_uniform_(self.hidden16.weight)
        # self.act16 = Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0, std=0.01)
                init.constant_(m.bias, 0)
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # output layer
        X = self.hidden3(X)
        X = self.act3(X)

        X = self.hidden4(X)
        X = self.act4(X)

        X = self.hidden5(X)
        X = self.act5(X)

        X = self.hidden6(X)
        X = self.act6(X)

        X = self.hidden7(X)
        X = self.act7(X)

        X = self.hidden8(X)
        X = self.act8(X)

        X = self.hidden9(X)
        X = self.act9(X)

        X = self.hidden10(X)
        X = self.act10(X)

        X = self.hidden11(X)
        X = self.act11(X)

        X = self.hidden12(X)
        X = self.act12(X)

        X = self.hidden13(X)
        X = self.act13(X)

        X = self.hidden14(X)
        X = self.act14(X)

        X = self.hidden15(X)
        X = self.act15(X)

        X = self.hidden16(X)
        # X = self.act16(X)

        return X




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

# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # b=next(iter(train))  non
    # prepare data loaders
    test_size = len(test)  # 获取测试集的大小 587
    print("test_size",test_size)
    #b=next(iter(train_dl))  non
    train_dl = DataLoader(train, batch_size=1, shuffle=True)
    test_dl = DataLoader(test, batch_size=test_size, shuffle=False)
    return train_dl, test_dl


# train the model
def train_model(train_dl, model):
    # define the optimization 交叉熵损失函数,用于计算模型的预测值和真实值之间的差异
     # criterion = CrossEntropyLoss()
    # optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = Adam(model.parameters())#adam自适应学习率的优化算法，它可以根据每个权重参数的梯度自适应地调整学习率
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            targets = targets.float()
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss

            loss = lossMME(yhat, targets)
            # credit assignment
            loss.backward()
            print("epoch: {}, batch: {}, loss: {}".format(epoch, i, loss.data))
            # update model weights
            optimizer.step()


# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = [], []
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()#张量 yhat 转换为 Numpy 数组。
        actual = targets.numpy()


        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc


# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat


# prepare the data
path1 = './data/MQ_train_1956_X_ok.csv'
path2= './data/MQ_train_1956_Y_ok.csv'
path='C:/Users/hydro1/lowen/color/data/trainXY.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network

model = MLP(207)
print(model)


# train the model
train_model(train_dl, model)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
# make a single prediction
row = [5.1, 3.5, 1.4, 0.2]
yhat = predict(row, model)
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))