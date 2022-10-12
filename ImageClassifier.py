# Permission to download dataset from internet
from pickletools import optimize
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
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Show data
#print(train_labels[0])
#print(train_images[0])

# Display the numpy array in grayscale
#plt.imshow(train_images[0], cmap='gray', vmin=0, vmax=255)
#plt.show()
#print(test_labels[0])

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
    keras.layers.Flatten(input_shape=(28, 28)),

    
#     Acrivation method is another layer of filtering which specifies the threshold
#     for what is good enough and what's not. 
#     relu = if anything is below 0 then return 0, else return the number it is, this
#     gets rid of negative numbers essentially
#     softmax = picks the greatest number out of everything. Each possible option has a 
#     probability when it reaches the output layer, softmax will return the one with 
#     the greatest number (the most likely answer) based on the pattern
    
    #Dense = every node in each column is connected to every other node in each column
    # Hidden layer is 128 deep, relu returns the value or 0 (works good enough, much faster)
    keras.layers.Dense(units=128, activation=tf.nn.relu),

    keras.layers.Dense(units=64, activation=tf.nn.relu),

    # Output layer is 0 - 10 (depending on the image), return maximum
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

"""
The loss function will tell us how correct / incorrect we are based on what the neural
net returns and what the actual image is. 
Optimizer makes changes to weights of connections between layers to improve the 
correctness.
"""
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train our model, using the training data
# epochs is the number of times the model should be trained on the sample data
model.fit(train_images, train_labels, epochs=5)

# Test our model using our testing data
test_loss = model.evaluate(test_images, test_labels)



plt.imshow(train_images[0], cmap='gray', vmin=0, vmax=255)
plt.show()

# Make predictions
predictions = model.predict(test_images)
# Print the predictions
print(predictions[0])
print(list(predictions[0]).index(max(predictions[0])))

# Print the correct label
print(test_labels[0])