# Permission to download dataset from internet
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Start of the code
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Load a pre-defined dataset
fashion_mnist = keras.datasets.fashion_mnist

# Pull out data from dataset
(train_images, train_labels), (test_images, trst_labels) = fashion_mnist.load_data()

# Show data
#print(train_labels[0])
#print(train_images[0])

# Display the numpy array in grayscale
plt.imshow(train_images[0], cmap='gray', vmin=0, vmax=255)
plt.show()

# Define our netural net structure
"""
Neural net is a bunch of different layers of nodes (called neurons) which are all connected.
In TensorFlow we want to design our model in a way that's compatible with out input data and
also the output data (a model is a neural net)

Keras allows us to define differnt graph structures, sequence of vertical columns that go in
a row. Each layer of our neural net is a vertical column of our data
        O
O               O
        O
O       
        O
O               O
        O
"""
model = keras.Sequential([
    # Input is a 28x28 image, flattened into a single 784x1 input layer
    keras.Layers.Flatten(input_shape=(28, 28)),
])