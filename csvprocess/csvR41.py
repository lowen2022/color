import pandas as pd
import csv
# # 读取CSV文件
file_path = 'HW-data_done.csv'
Datainput=64
Routput=31
# path2 ='MQ_test_488_X_ok.csv'
# df = pd.read_csv('MQ_train_1956_X_ok.csv')
# # 删除包含"T"的列
# df = df.drop(df.filter(like='T').columns, axis=1)
# df = df.drop(df.filter(like='Paper').columns, axis=1)
# # 保存修改后的数据到CSV文件
# df.to_csv('trainR84.csv', index=False)
path1 = 'HW-data_done.csv'
# df=pd.read_csv('LW104.csv')
# new_column_names = df.columns.tolist()
def RLABC104(path,new_column_names):

    # 读取CSV文件为DataFrame
    df = pd.read_csv(path)

    # 根据条件筛选要删除的行
    df=df[df['HW']!='Wavelength']
    df=df[df['HW']=='Value']
    df = df.drop(df.filter(like='HW').columns, axis=1)
    # 读取CSV文件并获取列名

    column_names = df.columns.tolist()
    # # 修改列名
    # new_column_names = ['NewCol1', 'NewCol2', 'NewCol3']  # 新的列名
    df.rename(columns=dict(zip(column_names, new_column_names)), inplace=True)

    # 保存修改后的数据到CSV文件
    df.to_csv('modified_file.csv', index=False)
path2 ='LW104.csv'
def R62(path):

    df = pd.read_csv(path)

    # 删除包含"labc"的列
    df = df.drop(df.filter(like='L1').columns, axis=1)
    df = df.drop(df.filter(like='a1').columns, axis=1)
    df = df.drop(df.filter(like='b1').columns, axis=1)
    df = df.drop(df.filter(like='L2').columns, axis=1)
    df = df.drop(df.filter(like='a2').columns, axis=1)
    df = df.drop(df.filter(like='b2').columns, axis=1)
    df = df.drop(df.filter(like='L3').columns, axis=1)
    df = df.drop(df.filter(like='a3').columns, axis=1)
    df = df.drop(df.filter(like='b3').columns, axis=1)
    df = df.drop(df.filter(like='C1').columns, axis=1)
    df = df.drop(df.filter(like='C2').columns, axis=1)

    # 将非数值列转换为数值类型
    df = df.apply(pd.to_numeric)

    # 将每个数据除以100
    df = df.apply(lambda x: x/100 )#if pd.api.types.is_numeric_dtype(x) else x
    # 保存修改后的数据到CSV文件
    df.to_csv('HWR93.csv', index=False)
# R62('modified_file.csv')
# RLABC104(path1,new_column_names)





# 读取第一个 CSV 文件的 L、A、B 列数据
df1 = pd.read_csv('./R2CIE/LW104.csv', usecols=['L1', 'a1', 'b1','L2', 'a2', 'b2','L3', 'a3', 'b3'])

# 读取第二个 CSV 文件的 L、A、B 列数据
df2 = pd.read_csv('./R2CIE/HW104.csv', usecols=['L1', 'a1', 'b1','L2', 'a2', 'b2','L3', 'a3', 'b3'])

# 创建一个新的 DataFrame，将两个文件的数据合并
df_new = pd.concat([df1, df2], ignore_index=True)

# 将合并后的数据写入新的 CSV 文件，列名为 "LAB"
df_new.to_csv('output.csv', index=False, header=['L1', 'a1', 'b1','L2', 'a2', 'b2','L3', 'a3', 'b3'])