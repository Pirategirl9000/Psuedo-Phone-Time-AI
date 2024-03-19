import tensorflow as tf
import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
import time 
import os

dataset = pd.readCSV('Phone_time_data.csv')

x = dataset.drop(columns=["age"])
y = dataset["age"]

x_train, x_test, y_train, y_test = train_test_split(dataset, test_size = 0.2, train_size = 0.8)

model = keras.models.Sequential()

model.add(keras.layers.Dense(256, x.shape[1:], activation = 'relu', max_value = 24.0))
model.add(keras.layers.Dense(256, activation = 'relu', max_value = 24.0))
model.add(keras.layers.Dense(1, activation = 'relu', max_value = 100.0))

model.fit(x_train, y_train, epochs = 500)

model.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics = ["accuracy"])

time.sleep(3)
os.system('cls')

print(model.predict(np.array([[1,3,0,0,1,0]])))






