import tensorflow as tf
import pandas as pd
import numpy as np
import keras

dataset = pd.readCSV('Phone_time_data.csv')

x = dataset.drop(columns=["age"])
y = dataset["age"]

