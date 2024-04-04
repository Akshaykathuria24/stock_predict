import pandas as pd
import numpy as np
import math
import datetime as dt
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

df = pd.read_csv('D:/Data Science/Projects/Current_project/stock_data.csv')
training_set = df.iloc[:,1:2].values
scaler = MinMaxScaler(feature_range = (0,1))
scaled_training_set = scaler.fit_transform(training_set)

x_train = []
y_train = []

for i in range(len(scaled_training_set)):
    x_train.append(scaled_training_set[i])
    y_train.append(scaled_training_set[i,0])
    
x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

