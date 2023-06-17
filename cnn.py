import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os



data_dir = "data"

data = os.listdir(os.path.join(data_dir, "1tl"))
print(data)


data = tf.keras.utils.image_dataset_from_directory("data")
