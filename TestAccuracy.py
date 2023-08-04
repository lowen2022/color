import torch
import torch.nn as nn
import pandas as pd
import numpy as np

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
#


# 定义 CIE XYZ 色度学函数
x_func = lambda w: 0.0001299 * w**(-1.61) if w >= 380 and w <= 780 else 0
y_func = lambda w: 0.0001299 * w**(-1.61) if w >= 380 and w <= 780 else 0
z_func = lambda w: 0.0001299 * w**(-1.61) if w >= 380 and w <= 780 else 0

# 定义参考白点的值
Xr = 0.9642
Yr = 1.0000
Zr = 0.8249

# 定义波长范围和反射率数据
wavelengths = np.arange(400, 701)  # 波长范围为 400nm 到 700nm
reflectances = np.random.rand(301)  # 随机生成 301 个反射率值

# 计算归一化因子 K
K = 100 / np.sum(reflectances * y_func(wavelengths))

# 计算 CIE XYZ 值
X = K * np.sum(reflectances * x_func(wavelengths))
Y = K * np.sum(reflectances * y_func(wavelengths))
Z = K * np.sum(reflectances * z_func(wavelengths))

# 将 CIE XYZ 值转换为 CIELAB 值
xr = X / Xr
yr = Y / Yr
zr = Z / Zr

if xr > 0.008856:
    fx = xr**(1/3)
else:
    fx = 7.787 * xr + 16/116
if yr > 0.008856:
    fy = yr**(1/3)
else:
    fy= 7.787 * yr + 16/116
if zr > 0.008856:
    fz = zr**(1/3)
else:
    fz = 7.787 * zr + 16/116

L = 116 * fy - 16
a = 500 * (fx - fy)
b = 200 * (fy - fz)

print(f"CIELAB: ({L:.3f}, {a:.3f}, {b:.3f})")
# 加载测试数据集和标签数据集
test_data = pd.read_csv('./data/MQ_test_488_X_ok.csv')
test_label = pd.read_csv('./data/MQ_test_488_Y_ok.csv')

# 将 DataFrame 转换为 NumPy 数组
test_data = test_data.to_numpy()
test_label = test_label.to_numpy()

# 将 NumPy 数组转换为 PyTorch 的 Tensor 对象
test_data = torch.from_numpy(test_data).float()
test_label = torch.from_numpy(test_label).long()

# 加载模型权重
model = SWPMPredictionModel()
model.load_state_dict(torch.load('model_epoch_40000.pth'))

# 将模型设置为评估模式
model.eval()

# 对测试数据集进行预测
with torch.no_grad():
    output = model(test_data)


# 计算准确度并输出结果
accuracy = (output.squeeze() == test_label.squeeze()).float().mean().item() * 100
print('Test accuracy: {:.2f}%'.format(accuracy))