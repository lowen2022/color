
import torch
import time
import csv
import torch.optim as optim
import pandas as pd
import visdom
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from tool import R2CIE
from tool import Ecalculate
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# 设置超参数
lr = 0.1
batch_size = 16

num_epochs = 1000
Datainput=31
Routput=3
path1='./csvprocess/R2CIE/Rtrain'
path2='./csvprocess/R2CIE/RtrainY'
path3='./csvprocess/R2CIE/Rtest.csv'
path4='./csvprocess/R2CIE/RtestY.csv'


# 读取训练数据和标签数据
train_data = pd.read_csv(path3)#path1 or path5
label_data = pd.read_csv(path4)

# # 创建一个 StandardScaler 对象
# scaler = StandardScaler()
#
# # 对训练数据进行零均值和方差为1的标准化
# train_data_scaled = scaler.fit_transform(train_data)
#
# # 将标准化后的数据重新转换为 DataFrame
# train_data = pd.DataFrame(train_data_scaled, columns=train_data.columns)
#

# 将训练数据和标签数据转换为 NumPy 数组
train_data_array = train_data.values[:64]
train_label_array = label_data.values[:64]

# 将 NumPy 数组转换为 PyTorch 张量
train_data_tensor = torch.tensor(train_data_array, dtype=torch.float32)
train_label_tensor = torch.tensor(train_label_array, dtype=torch.float32)


# 创建数据集
dataset = torch.utils.data.TensorDataset(train_data_tensor, train_label_tensor)

# # 创建采样器来打乱数据集
# indices = list(range(len(dataset)))
# np.random.shuffle(indices)

# 使用采样器创建数据加载器，同时打乱数据集并保持标签对应关系
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),drop_last=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True,drop_last=True)



# 读取训练数据和标签数据
valid_data = pd.read_csv(path3)[64:]#path3(R207) or path6(R84)
valid_label = pd.read_csv(path4)[64:]

# 将训练数据和标签数据转换为 NumPy 数组
valid_data_array = valid_data.values
valid_label_array = valid_label.values

# 将 NumPy 数组转换为 PyTorch 张量
valid_data_tensor = torch.tensor(valid_data_array, dtype=torch.float32)
valid_label_tensor = torch.tensor(valid_label_array, dtype=torch.float32)


# 创建数据集

datasetvalid = torch.utils.data.TensorDataset(valid_data_tensor, valid_label_tensor)
dataloadervalid = torch.utils.data.DataLoader(datasetvalid, batch_size=batch_size)

# 定义MLP模型类
# 定义模型 and loss
def lossMME(y_pred,y_true):

    m = y_true.shape[0]
    mu = torch.mean(y_true)

    mape = torch.sum(torch.abs((y_true - y_pred) / y_true))/31
    mae = torch.sum(torch.abs(y_true - y_pred))/31

    Var_all = torch.sum(torch.var(y_true - y_pred))/31

    alpha = 3
    beta = 2

    loss = mape+alpha * mae + beta * Var_all  #+0.000015 * torch.norm(y_pred, p=1) + 0.000003 * torch.norm(y_pred, p=2)

    return loss
def lossE(y_pred,y_true):
    # 确保两个张量具有相同的形状
    assert y_pred.shape == y_true.shape, "两个张量的形状不匹配"

    # 计算平方差之和
    squared_diff = torch.square(y_pred - y_true)
    sum_of_squared_diff = torch.sum(squared_diff)

    return sum_of_squared_diff
class SWPMPredictionModel(nn.Module):
    def __init__(self,Datainput,Routput):
        super(SWPMPredictionModel, self).__init__()
        self.fc1 = nn.Linear(Datainput, 22)
        self.fc2 = nn.Linear(22,22)
        # self.fc3 = nn.Linear(166, 166)
        # self.fc4 = nn.Linear(166,166)
        # self.fc5 = nn.Linear(166, 166)
        self.fcoutput = nn.Linear(22,Routput)
        self.relu= nn.ReLU()
        self.leakyrelu=nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(0.5)  # 添加 dropout 层，丢弃概率为 0.5
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid=nn.Sigmoid()


    def forward(self, x):
        x = self.fc1(x)
        x = self.leakyrelu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        # # x = self.leakyrelu(x)
        # x = self.fc3(x)
        # x = self.leakyrelu(x)
        # x = self.fc4(x)
        # x = self.relu(x)
        # # x = self.dropout(x)
        # x = self.fc5(x)
        x = self.relu(x)
        x = self.fcoutput(x)
        # x = self.sigmoid(x)
        #Relu and sigmoid for output


        return x

# 初始化模型、损失函数和优化器
# visdom to see the graph of loss
# 连接到Visdom服务器
vis = visdom.Visdom(server='http://localhost', port=8097)

# 创建窗口用于可视化训练损失和验证损失
loss_win = 'loss'
vis.line(X=np.array([0]), Y=np.array([0]), win=loss_win, opts=dict(title='Train Loss'))

# 创建窗口用于可视化训练准确率和验证准确率
accuracy_win = 'accuracy'
vis.line(X=np.array([0]), Y=np.array([0]), win=accuracy_win, opts=dict(title='Train Accuracy'))

# 创建窗口用于可视化验证损失
vloss_win = 'vloss'
vis.line(X=np.array([0]), Y=np.array([0]), win=vloss_win, opts=dict(title='Valid Loss'))

# 创建窗口用于可视化验证准确率
vaccuracy_win = 'vaccuracy'
vis.line(X=np.array([0]), Y=np.array([0]), win=vaccuracy_win, opts=dict(title='Valid Accuracy'))

model = SWPMPredictionModel(Datainput,Routput )
# criterion=lossMME
criterion =lossE
nn.init.xavier_uniform_(model.fc1.weight)
nn.init.zeros_(model.fc1.bias)
nn.init.xavier_uniform_(model.fc2.weight)
nn.init.zeros_(model.fc2.bias)
# nn.init.xavier_uniform_(model.fc3.weight)
# nn.init.zeros_(model.fc3.bias)
# nn.init.xavier_uniform_(model.fc4.weight)
# nn.init.zeros_(model.fc4.bias)
# nn.init.xavier_uniform_(model.fc5.weight)
# nn.init.zeros_(model.fc5.bias)
nn.init.xavier_uniform_(model.fcoutput.weight)
nn.init.zeros_(model.fcoutput.bias)


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.85, patience=30, threshold=1e-4)
# 定义预热的epoch数
warmup_epochs = 5
#
scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.3)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-3)
parametertype=f'{batch_size}_{lr}'
# 训练模型
for epoch in range(num_epochs):
    # 更新学习率
    if epoch < warmup_epochs:
        # 预热阶段，按预定规则逐渐增加学习率
        warmup_factor = min((epoch + 1) / warmup_epochs, 1.0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * warmup_factor
    # else:
        # 根据epoch调整学习率
        # scheduler.step()
    # 训练模式
    model.train()
    train_loss = 0.0
    train_correct = 0

    # 批量训练
    for inputs, labels in dataloader :
        E=[]
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # loss=criterion(outputs,labels)
        # vis.line(X=np.array([epoch]), Y=np.array([loss_detached]), win='loss', update='append')
        loss_detached=loss.detach().numpy()

        if(loss_detached<=5):
                train_correct+=1
        E.append(loss_detached)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算训练损失和准确率
        train_loss += loss.item() * inputs.size(0)

        outputs_array = outputs.detach().numpy()
        # for result, tgt in zip(outputs_array, labels):
        #     L1, a1, b1 = R2CIE(result, Routput)
        #     L2, a2, b2 = R2CIE(tgt, Routput)
        #     e = Ecalculate(L1, a1, b1, L2, a2, b2)
        #     if(e<=5):
        #         train_correct+=1
        #
        #     E.append(e)


    # 打印学习率
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
        print("Current learning rate:", current_lr)
    for name, parms in model.named_parameters():
        print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
              ' -->grad_value:', torch.mean(parms.grad))
    if epoch % 100 == 0:
        # Emean=np.array(E)
        # print('Emean',Emean)
        current_time = time.strftime('%m-%d__%H:%M:%S')
        print('Epoch {}, Loss: {} , Time: {}'.format(epoch, loss.item(), current_time))
        # for name, parms in model.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
        #           ' -->grad_value:', torch.mean(parms.grad))
        torch.save(model.state_dict(), 'epoch_{}_{}.pth'.format(epoch, parametertype))


    # 验证模式
    model.eval()
    valid_loss = 0.0
    valid_correct = 0

    # 禁用梯度计算
    with torch.no_grad():
        for validinputs, valid_label in dataloadervalid:
            validoutput = model(validinputs)  # tensor(488x41)

            for result, tgt in zip(validoutput, valid_label):
                E=[]
                val_loss = criterion(result, tgt)
                val_loss_detach =val_loss.detach().numpy()

                # E.append(evalid)
                if(val_loss_detach<=5):
                    valid_correct+=1
                # Emean = np.array(E)


                # 计算验证损失和准确率
                valid_loss += val_loss.item() * inputs.size(0)
        # 计算平均损失和准确率
        train_loss = train_loss / len(train_data)
        valid_loss = valid_loss / len(valid_data)
        train_accuracy = train_correct / len(train_data)
        valid_accuracy = valid_correct / len(valid_data)
        if(epoch!=0):
            vis.line(X=np.array([epoch]), Y=np.array([train_loss]), win=loss_win, update='append')
            vis.line(X=np.array([epoch]), Y=np.array([train_accuracy]), win=accuracy_win, update='append')
            vis.line(X=np.array([epoch]), Y=np.array([valid_loss]), win=vloss_win, update='append')
            vis.line(X=np.array([epoch]), Y=np.array([valid_accuracy]), win=vaccuracy_win, update='append')


        # 输出训练过程中的相关指标
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}')
        print(f'Valid Loss: {valid_loss:.4f} | Valid Accuracy: {valid_accuracy:.4f}')