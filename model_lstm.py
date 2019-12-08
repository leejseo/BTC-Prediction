import numpy as np
import pandas as pd

from math import sqrt

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import time
import datetime

import matplotlib.pyplot as plt

df_train = pd.read_csv("train.csv", index_col = "time")
df_test = pd.read_csv("test.csv", index_col = "time")

num_back = 5

num_train = len(df_train)

scaler = MinMaxScaler(feature_range=(0, 100))
np_arr = np.concatenate((np.array(df_train), np.array(df_test)), axis=0)
np_norm = scaler.fit_transform(np_arr)

df_train = np_norm[:num_train]
df_test = np_norm[num_train:]

def create_dataset(df):
    X, y = [], []
    for i in range(len(df)-num_back-1):
        X.append(df[i:i+num_back])
        y.append(df[i+num_back])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(df_train)
X_test, y_test = create_dataset(df_test)

X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
y_train.reshape(y_train.shape[0])

X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
y_test.reshape(y_test.shape[0])

# Simple LSTM model

model = Sequential()
model.add(LSTM(units=3, input_shape=X_train[0].shape))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=40, batch_size=1)

y_predict = model.predict(X_test)
y_predict = scaler.inverse_transform(y_predict)
y_test = scaler.inverse_transform(y_test)
score = sqrt(mean_squared_error(y_test, y_predict))

print ("Score: %.2f mse" %score)

plt.plot(y_predict)
plt.plot(y_test)
plt.show()