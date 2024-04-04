import pandas as pd
import numpy as np
import math
import datetime as dt
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('D:/Data Science/Projects/Current_project/stock_data.csv')
training_set = df.iloc[:,1:2].values
scaler = MinMaxScaler(feature_range = (0,1))
scaled_training_set = scaler.fit_transform(training_set)
x_train = []
y_train = []
for i in range(1,0):
    x_train.append(scaled_training_set[i-60:i, 0])
    y_train.append(scaled_training_set[i,0])
X_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)
print(y_train.shape)
# x_train = np.array(x_train)
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# print(x_train.shape)
