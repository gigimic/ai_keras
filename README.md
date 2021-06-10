Understanding Keras
--------------------

Keras makes coding deep neural networks simpler.

create sequential model in keras:

from keras.models import Sequential
model = Sequential() 

The keras.models.Sequential class is a wrapper for the neural network model that treates the network as a 
sequence of layers. It implements the keras model interface with common methods like 
compile(), fit() and evaluate(). 

Layers:
Keras Layer class provides a common interface for a variety of standard network layers. These are fully connected layers, max pool layers, activation layers etc. A Layer can be added using add() method.

keras requires the input shape to be specified for the first layer and it will automatically infer the shape of all other layers. 

The first hidden layer (eg. model.add(Dense(32, input_dim=X.shape[1]))) creates 32 nodes which each expect to receive 2-element vectors as inputs. 

Each layer takes the output from the previous layer as inputs and pipes through to the next layer. 

The activation layers in keras are equivalent to specifying an activation function in the Dense layers 
model.add(Dense(128))
model.add(Activation('softmax'))
or
model.add(Dense(128, activation='softmax'))

Compiling the model:
The model binds the optimizer, loss function and other parameters required before the model can run.
loss function - categorical_crossentropy (applicable when there are only two classes)
optimizer -adam (when speed is the priority)
matrix to evaluate the model with - accuracy

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])

model.summary()

model can be trained with the fit() method
model.fit(X, y, nb_epoch=1000, verbose=0)

to evaluate the model
model.evaluate()


https://keras.io 