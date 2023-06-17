import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import json



# configs
data_dir = "data"

with open("coin_desc.json") as f:
    class_desc = json.load(f)
print(class_desc)

# data pipleline
data = tf.keras.utils.image_dataset_from_directory("data", batch_size=24)
data = data.map(lambda x, y: (x/255, y)) # scale
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()


# images represented as numpy array
print("len is:", len(batch))
print(batch[1])
print("data size", len(data))


# split data
train_size = int(len(data) * .7)
val_size = int(len(data) * .2)
test_size = int(len(data) * .1)


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip( train_size + val_size).take(test_size)



print(train_size, val_size, test_size)
