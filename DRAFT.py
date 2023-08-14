import colour
import numpy as np
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
from sklearn.preprocessing import StandardScaler
import csv
batch_size=32
path1 = './data/MQ_train_1956_X_ok.csv'
path2= './data/MQ_train_1956_Y_ok.csv'
path='C:/Users/hydro1/lowen/color/data/trainXY.csv'
# 读取训练数据和标签数据
train_data = pd.read_csv(path1)[:256]
label_data = pd.read_csv(path2)[:256]

# 创建一个 StandardScaler 对象
scaler = StandardScaler()

# 对训练数据进行零均值和方差为1的标准化
train_data_scaled = scaler.fit_transform(train_data)

# 将标准化后的数据重新转换为 DataFrame
train_data_scaled = pd.DataFrame(train_data_scaled, columns=train_data.columns)

# 将训练数据和标签数据转换为 NumPy 数组
train_data_array = train_data_scaled.values
train_label_array = label_data.values

# 将 NumPy 数组转换为 PyTorch 张量
train_data_tensor = torch.tensor(train_data_array, dtype=torch.float32)
train_label_tensor = torch.tensor(train_label_array, dtype=torch.float32)

# #零中心化Zero-centering    variance=1
# train_data_mean = torch.mean(train_data_tensor, dim=0)
# train_data_centered = train_data_tensor - train_data_mean
# train_data_centerd_variance=torch.var(train_data_centered)
# train_data_centered /=torch.sqrt(train_data_centerd_variance)
# 创建数据集
dataset = torch.utils.data.TensorDataset(train_data_tensor, train_label_tensor)

# 创建采样器来打乱数据集
indices = list(range(len(dataset)))
np.random.shuffle(indices)

# 使用采样器创建数据加载器，同时打乱数据集并保持标签对应关系
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),drop_last=True)

for inputs, labels in dataloader :
    inputs=inputs.numpy()
    trainmean=np.mean(inputs,axis=1)
    print(trainmean)
# with open("aftercenter.csv", 'w', newline='') as f:
#         writer = csv.writer(f)
#         for row in train_data_centered:
#
#             writer.writerow(row.tolist())
