import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



test_data = pd.read_csv('./data/MQ_train_1956_X_ok.csv',nrows=100)
test_label= pd.read_csv('./data/MQ_train_1956_Y_ok.csv',nrows=100)

print(test_data)



