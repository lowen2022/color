import pandas
import csv

# with open('data/trainXY.csv','r') as f:
#     reader=csv.reader(f)
#     next(reader)
#     r=next(reader)
#     data=[float(x) for x in r]
# row = data[:-41]
# # self.X = df.values[:, :-41]
# print(row)
#====================
import numpy as np
import pandas as pd
import colour
import numpy as np

def calculate(a,b):
    return [x*y for x,y in zip(a,b)]

# 定义 x, y, z 函数  380-780  41
x_bar = np.array(
        [0.001368, 0.004243, 0.014310, 0.043510, 0.134380, 0.2839, 0.34828, 0.3362, 0.2908, 0.19536, 0.09564, 0.03201,
         0.0049, 0.0093, 0.06327, 0.1655, 0.2904, 0.43345, 0.5945, 0.7621, 0.9163, 1.0263, 1.0622, 1.0026, 0.8544,
         0.6424, 0.4479, 0.2835, 0.1649, 0.0874, 0.0468, 0.0227, 0.0114, 0.0058, 0.0029, 0.0014, 0.0007, 0.0003, 0.0002,
         0.0001, 0.000042])
y_bar = np.array(
    [0.0000, 0.0001, 0.0004, 0.0012, 0.0040, 0.0116, 0.0230, 0.0380, 0.0600, 0.0910, 0.1390, 0.2080, 0.3230, 0.5030,
     0.7100, 0.8620, 0.9540, 0.9950, 0.9950, 0.9520, 0.8700, 0.7570, 0.6310, 0.5030, 0.3810, 0.2650, 0.1750, 0.1070,
     0.0610, 0.0320, 0.0170, 0.0082, 0.0041, 0.0021, 0.001047, 0.00052, 0.000249, 0.00012, 0.00006, 0.00003, 0.000015])
z_bar = np.array(
    [0.0065, 0.0201, 0.0679, 0.2074, 0.6456, 1.3856, 1.7471, 1.7721, 1.6692, 1.2876, 0.8130, 0.4652, 0.2720, 0.1582,
     0.0782, 0.0422, 0.0203, 0.0087, 0.0039, 0.0021, 0.0017, 0.0011, 0.0008, 0.0003, 0.0002, 0.0002, 0.0001, 0.0001,
     0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0])



# 定义 f 函数




def f(t):
    delta = 6/29
    if t > delta**3:
        return t**(1/3)
    else:
        return (1/3)*(t/(delta**2)) + (4/29)
# 假设 wavelength 值是等间隔的


# 假设反射率数据保存在名为 "reflectances.csv" 的 csv 文件中  31
data = np.genfromtxt('data/colortest.csv', delimiter=',')


# 获取 reflectance 列
reflectance = data[:]

# D65 光源标准辐射强度数据，41个数据点   380-780
D65_values = [49.975,54.648,82.754,91.486,93.431,86.682,104.865,117.08,117.812,114.861,115.923,108.811,109.354,107.802,104.790,107.689,104.405,104.046,100,96.334,95.788,88.685,90.0062,89.599,87.698,83.288,83.699,80.026,80.214,82.2778,78.284,69.721,71.609,74.349,61.604,69.8856,75.087,63.592,46.418,66.805,63.3828]

# 归一化反射率数据

with open('data/white.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # 获取标题行
    second_row = next(reader)  # 获取第二行数据
    third_row = next(reader)  # 获取第三行数据
white=[x for x in third_row if x!=""]#31  300-700
with open('data/black.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # 获取标题行
    second_row = next(reader)  # 获取第二行数据
    third_row = next(reader)  # 获取第三行数据
black=[x for x in third_row if x!=""]#31  400-700


print("white,black")
print(white)
print(black)
wave=[None]*31
for i in range(31):
    wave[i]=float(white[i])-float(black[i])
epsilon1 = 1e-9  # 定义一个小的正数
reflectance_normalized = []
max_reflectance = max(reflectance)
for i in range(len(reflectance)):
    reflectance_normalized.append(reflectance[i]  / (wave[i]+epsilon1))

print(reflectance_normalized)
# with open('result.txt','w') as f:
#     for i in reflectance_normalized:
#         f.write(str(i))
# f.close()
# 计算对于反射率数据，需要将其转换为 XYZ 颜色空间中的 tristimulus 值，然后再将其转换为 CIELAB 空间中的 Lab 值。可以使用以下代码来实现：


# 计算 tristimulus 值
D65_1931X=calculate(D65_values,x_bar)
D65_1931Y=calculate(D65_values,y_bar)
D65_1931Z=calculate(D65_values,z_bar)
X = np.sum(calculate(D65_1931X[2:33],reflectance_normalized))
Y = np.sum(calculate(D65_1931Y[2:33],reflectance_normalized))
Z = np.sum(calculate(D65_1931Z[2:33],reflectance_normalized))
X=X*0.1
Y=Y*0.1
Z=Z*0.1
print("X,Y,Z",X,Y,Z)

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
L = 116*f(Y/Yn) - 16
a = 500*(f(X/Xn) - f(Y/Yn))
b = 200*(f(Y/Yn) - f(Z/Zn))

lab_values = L, a, b

def Ecalculate(L1,a1,b1,L2,a2,b2):
    E=(L1-L2)**2+(a1-a2)**2+(b1-b2)**2
    E=E**0.5
    return E

E=Ecalculate(L,a,b,78,27,1)
print('E',E)
# 输出结果
print("L: ", L)
print("a: ", a)
print("b: ", b)