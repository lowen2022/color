import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import csv
# from model_v3 import SWPMPredictionModel
# 定义模型
# 定义模型

#
Datainput=64
Routput=31
def R2CIE(R,Rnumber):
    # 假设反射率数据保存在名为 "reflectances.csv" 的 csv 文件中  31
    # data = np.genfromtxt('data/colortest.csv', delimiter=',')
    # data = np.genfromtxt(Rpath, delimiter=',')
    #second way:directly read R np
    data=R
    # 获取 reflectance 列
    reflectance = data[:]

    def calculate(a, b):
        return [x * y for x, y in zip(a, b)]
    # 定义 f 函数
    def f(t):
        delta = 6 / 29
        if t > delta ** 3:
            return t ** (1 / 3)
        else:
            return (1 / 3) * (t / (delta ** 2)) + (4 / 29)
    # 定义 x, y, z 函数  380-780  41
    x_bar = np.array([0.001368, 0.004243, 0.014310, 0.043510, 0.134380, 0.2839, 0.34828, 0.3362, 0.2908, 0.19536, 0.09564, 0.03201,
         0.0049, 0.0093, 0.06327, 0.1655, 0.2904, 0.43345, 0.5945, 0.7621, 0.9163, 1.0263, 1.0622, 1.0026, 0.8544,
         0.6424, 0.4479, 0.2835, 0.1649, 0.0874, 0.0468, 0.0227, 0.0114, 0.0058, 0.0029, 0.0014, 0.0007, 0.0003, 0.0002,
         0.0001, 0.000042])
    y_bar = np.array([0.0000, 0.0001, 0.0004, 0.0012, 0.0040, 0.0116, 0.0230, 0.0380, 0.0600, 0.0910, 0.1390, 0.2080, 0.3230, 0.5030,
         0.7100, 0.8620, 0.9540, 0.9950, 0.9950, 0.9520, 0.8700, 0.7570, 0.6310, 0.5030, 0.3810, 0.2650, 0.1750, 0.1070,
         0.0610, 0.0320, 0.0170, 0.0082, 0.0041, 0.0021, 0.001047, 0.00052, 0.000249, 0.00012, 0.00006, 0.00003,
         0.000015])
    z_bar = np.array([0.0065, 0.0201, 0.0679, 0.2074, 0.6456, 1.3856, 1.7471, 1.7721, 1.6692, 1.2876, 0.8130, 0.4652, 0.2720, 0.1582,
         0.0782, 0.0422, 0.0203, 0.0087, 0.0039, 0.0021, 0.0017, 0.0011, 0.0008, 0.0003, 0.0002, 0.0002, 0.0001, 0.0001,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0])
    # D65 光源标准辐射强度数据，41个数据点   380-780
    D65_values = [49.975, 54.648, 82.754, 91.486, 93.431, 86.682, 104.865, 117.08, 117.812, 114.861, 115.923, 108.811,
                  109.354, 107.802, 104.790, 107.689, 104.405, 104.046, 100, 96.334, 95.788, 88.685, 90.0062, 89.599,
                  87.698, 83.288, 83.699, 80.026, 80.214, 82.2778, 78.284, 69.721, 71.609, 74.349, 61.604, 69.8856,
                  75.087, 63.592, 46.418, 66.805, 63.3828]


    # 计算对于反射率数据，需要将其转换为 XYZ 颜色空间中的 tristimulus 值，然后再将其转换为 CIELAB 空间中的 Lab 值。可以使用以下代码来实现：
    reflectance_normalized = reflectance
    # 计算 tristimulus 值
    D65_1931X = calculate(D65_values, x_bar)
    D65_1931Y = calculate(D65_values, y_bar)
    D65_1931Z = calculate(D65_values, z_bar)
    if(Rnumber==31):
        X = np.sum(calculate(D65_1931X[2:33], reflectance_normalized))
        Y = np.sum(calculate(D65_1931Y[2:33], reflectance_normalized))
        Z = np.sum(calculate(D65_1931Z[2:33], reflectance_normalized))
    else:#380-780 41
        X = np.sum(calculate(D65_1931X, reflectance_normalized))
        Y = np.sum(calculate(D65_1931Y, reflectance_normalized))
        Z = np.sum(calculate(D65_1931Z, reflectance_normalized))
    X = X * 0.1
    Y = Y * 0.1
    Z = Z * 0.1
    # print("X,Y,Z", X, Y, Z)

    # # 对 XYZ 值进行 gamma 校正  can't!   L a will be much smaller
    # epsilon = 0.008856
    # kappa = 903.3
    # X = X**(1/3) if X > epsilon else (kappa*X + 16)/116
    # Y = Y**(1/3) if Y > epsilon else (kappa*Y + 16)/116
    # Z = Z**(1/3) if Z > epsilon else (kappa*Z + 16)/116

    # 将 tristimulus 值转换为 Lab 值
    Xn = 95.047
    Yn = 100.000
    Zn = 108.883

    # 计算 L*, a* 和 b* 值
    L = 116 * f(Y / Yn) - 16
    a = 500 * (f(X / Xn) - f(Y / Yn))
    b = 200 * (f(Y / Yn) - f(Z / Zn))

    lab_values = L, a, b

    # def Ecalculate(L1, a1, b1, L2, a2, b2):
    #     E = (L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2
    #     E = E ** 0.5
    #     return E
    #
    # E1 = Ecalculate(L, a, b, 78, 27, 1)
    # E2 = Ecalculate(L, a, b, 78, 30, 1)
    # print('E1_codetest_tgt', E1)
    # print('E2_codetest_online', E2)
    # 输出结果
    print("L: ", L)
    print("a: ", a)
    print("b: ", b)
    return L,a,b
def accuracy(outputs,test_label):
    ACC=0
    numall=0
    for result,tgt in zip(outputs,test_label):
        L1,a1,b1=R2CIE(result,41)
        L2,a2,b2=R2CIE(tgt,41)
        # print("L1,a1,b1",L1,a1,b1)
        # print("L2,a2,b2", L2, a2, b2)
        E=(Ecalculate(L1, a1, b1, L2, a2, b2))
        numall+=1
        if(E<=10):
            ACC+=1
    return(ACC/numall)

def Ecalculate(L1, a1, b1, L2, a2, b2):
    E = (L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2
    E = E ** 0.5
    print("E",E)
    return E
    # E1 = Ecalculate(L, a, b, 78, 27, 1)
    # E2 = Ecalculate(L, a, b, 78, 30, 1)
    # print('E1_codetest_tgt', E1)
    # print('E2_codetest_online', E2)

# 加载测试数据集和标签数据集

path1='../csvprocess/before_n.csv'
path2='./csvprocess/Hydro64Y_n.csv'
path3='../csvprocess/LW104.csv'
def read_csv_columns(filename, columns):
    df = pd.read_csv(filename, usecols=columns)
    return df

c1_lab = ['L1', 'a1', 'b1']  # 需要读取的列名
c2_lab = ['L2', 'a2', 'b2']
c3_lab = ['L3', 'a3', 'b3']
c1_lab = read_csv_columns(path3, c1_lab)
c2_lab = read_csv_columns(path3, c2_lab)
c3_lab = read_csv_columns(path3, c3_lab)

c1 = pd.read_csv(path1, usecols=range(31)).values
c2 = pd.read_csv(path1, usecols=range(31, 62)).values
c3 = pd.read_csv(path1, usecols=range(62, 93)).values
E=[]
for i in range(len(c1_lab)):
    # 提取每行的列数据
    row1 = c1_lab.iloc[i]
    row2 = c2_lab.iloc[i]
    row3 = c3_lab.iloc[i]

    L11, a11, b11 = row1['L1'], row1['a1'], row1['b1']
    L22, a22, b22 = row2['L2'], row2['a2'], row2['b2']
    L33, a33, b33 = row3['L3'], row3['a3'], row3['b3']
    a = c1[i]
    b = c2[i]
    c = c3[i]

    L1, a1, b1 = R2CIE(a, 31)
    L2, a2, b2 = R2CIE(b, 31)
    L3, a3, b3 = R2CIE(c, 31)

    e1 = Ecalculate(L1, a1, b1, L11, a11, b11)
    e2 = Ecalculate(L2, a2, b2, L22, a22, b22)
    e3 = Ecalculate(L3, a3, b3, L33, a33, b33)
    E.append((e1,e2,e3))
    Emean=np.mean(E)
print('Emean',Emean)#Emean 6.7291875240399985




