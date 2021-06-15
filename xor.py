
'''A simple multi-layer feedforward neural network to solve the XOR problem.

Set the first layer to a Dense() layer with an output width of 8 nodes and 
the input_dim set to the size of the training samples (in this case 2).
Add a tanh activation function.
Set the output layer width to 1, since the output has only two classes. 
(We can use 0 for one class and 1 for the other)
Use a sigmoid activation function after the output layer.
Run the model for 50 epochs.
'''

import numpy as np
from keras.utils import np_utils
import tensorflow as tf
# Using TensorFlow 1.0.0; use tf.python_io in later versions
# tf.python.control_flow_ops = tf
tf.python_io = tf

# Set random seed
np.random.seed(42)

# data
X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

# One-hot encoding the output
y = np_utils.to_categorical(y)

# Building the model
xor = Sequential()
xor.add(Dense(32, input_dim=2))
xor.add(Activation("tanh"))
xor.add(Dense(2))
xor.add(Activation("sigmoid"))

xor.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ['accuracy'])

# print model architecture
print(xor.summary())

# Fitting the model
history = xor.fit(X, y, epochs=3, verbose=0)

# Scoring the model
score = xor.evaluate(X, y)
print("\nAccuracy: ", score[-1])

# Checking the predictions
print("\nPredictions:")
print(xor.predict(X))