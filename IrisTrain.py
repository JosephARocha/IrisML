import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import to_categorical
import numpy as np
import pandas as pd

# Read In Iris.data into an array
dataset = pd.read_csv('Iris.data')

#The y values are text, we turn them into integers to feed into the network
dataset = dataset.replace("Iris-setosa", "0")
dataset = dataset.replace("Iris-versicolor", "1")
dataset = dataset.replace("Iris-virginica", "2")

# columns 0-4 are our features (x values), column 5 is our y value... f(x1,x2,x3,x4) = y
x_train = dataset.iloc[:,0:4]
y_train = dataset.iloc[:,4:5]
#Turn the y values into a matrix [0,0,1] [0,1,0] [1,0,0]
y_train = to_categorical(y_train)

#simple sequential model
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=4))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax'))

#https://en.wikipedia.org/wiki/Stochastic_gradient_descent
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Train our model for 100 iteration of the training data
model.fit(x_train, y_train, epochs=100, shuffle=True,validation_split=0.2)
