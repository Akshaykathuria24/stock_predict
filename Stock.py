import pandas as pd
import numpy as np
import math
import datetime as dt
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Read the dataset
df = pd.read_csv('D:/Data Science/Projects/Current_project/stock_data.csv')
training_set = df.iloc[:, 1:2].values

# Scale the training set
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_training_set = scaler.fit_transform(training_set)

# Prepare training data for LSTM
x_train = []
y_train = []
for i in range(len(scaled_training_set)):
    x_train.append(scaled_training_set[i])
    y_train.append(scaled_training_set[i, 0])
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Define and train the LSTM model
regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(x_train, y_train, epochs=1, batch_size=32)

# Prepare test data for prediction
dataset_total = pd.concat((df['open'], df['open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(df) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)
x_test = []
for i in range(60, 90):
    x_test.append(inputs[i - 60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict stock prices
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Plot the results
plt.plot(df.index, training_set, color='red', label='Current Stock Price')
plt.plot(df.index[-len(predicted_stock_price):], predicted_stock_price, color='blue', label='Predicted Stock Price')
plt.title('Stock Price Prediction By Akshay Kathuria')
plt.xlabel('Time (Years)')
plt.ylabel('Price')
plt.legend()
plt.show()