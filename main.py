import tensorflow
import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
import time

dataset = pd.read_csv('Phone_time_data.csv')

x = dataset.drop(columns=["age"])
y = dataset["age"]

x_train, x_test, y_train, y_test = train_test_split(x, y)

model = keras.models.Sequential()

model.add(keras.layers.Dense(256, input_shape = x.shape[1:], activation = 'linear'))
model.add(keras.layers.Dense(256, activation = 'linear'))
model.add(keras.layers.Dense(1, activation = 'linear'))

model.compile(optimizer = 'adam', loss = "mean_absolute_error", metrics = ["accuracy"])
model.fit(x_train, y_train, epochs = 500)

time.sleep(3)

print(model.predict(np.array([[1,3,0,0,1,0]])))






