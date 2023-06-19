import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout



# configs
data_dir = "data"


# data pipleline
data = tf.keras.utils.image_dataset_from_directory("data", batch_size=24)
data = data.map(lambda x, y: (x/255, y)) # scale
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()


# images represented as numpy array
# print("len is:", len(batch))
# print(batch[1])
# print("data size", len(data))


# split data
train_size = int(len(data) * .7)
val_size = int(len(data) * .2)
test_size = int(len(data) * .1)


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip( train_size + val_size).take(test_size)


# deep learning model

model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

model.summary()


# train
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=5, validation_data=val, callbacks=[tensorboard_callback])

# evalutions

# save the model
model.save(os.path.join('','coin_classifier.h5'))
