"""
Title: Neural Networks: Fully Connected
Author: Rodrigo Rangel
Description: * Neural Networks are built in a way that mimic the human brain
             * They 'activate' neurons based on activation function, where the signal
               strength is determined by the weights that connect each layer
             * To go from one layer to the next, the dot product is computer between the
               current layer and the next layer, afterwards the activation function is applied
             * The process is called forward propagation
             * The next step is backwards propagation, where the partial derivative of the loss function
               is taken in order to modify the weights
             * Finally the model iterates through a series of forward and backward propagations, called
               gradient decent
             * The architecure of basic Nerual Networks is as follows:
                 1. Input Layer
                 2. Hidden Layer(s)
                 3. Output Layer

             * Neural Networks can be used for:
                 - Binary Class
                 - Multi-Class
                 - Regression
"""

#-----------------------------------------------------------------------------#
#                              Dependencies                                   #
#-----------------------------------------------------------------------------#
# Dependencies
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils

# Creating Dataset
X,y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)

X2,y2 = make_regression(n_samples=1000, n_features=4,
                           n_informative=2,
                           random_state=0, shuffle=False)

"""
Notes:
Neural Networks perform best when the data is normlized
"""

#-----------------------------------------------------------------------------#
#                          Neural Network: Binary Class                       #
#-----------------------------------------------------------------------------#

# Initializing Model
model = Sequential()

# Input Layer & Hidden Layer 1
model.add(Dense(100, activation='relu',input_shape=(4,)))

# Hidden Layer 2
model.add(Dense(50, activation='relu'))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

# Compile Parameters
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

# Train Model
model.fit(X, y, epochs=20, batch_size=32)

"""
Notes:
Input Layer: 4 Neurons
Hidden Layer 1: 100 Neurons Relu Activation
Hidden Layer 2: 50 Neurons Relu Activation
Output Layer 1: 1 Neurons Sigmoid Activation
Compiler Loss: Binary Cross-Entropy
"""
#-----------------------------------------------------------------------------#
#                          Neural Network: Multi- Class                       #
#-----------------------------------------------------------------------------#

# Generating Dummy Data (5 Classes)
import numpy as np
X = np.random.random((1000, 3))
y = np.random.randint(5, size=(1000, 1)) # 5 Classes

# Initializing Model
model = Sequential()

# Input Layer & Hidden Layer 1
model.add(Dense(32, activation='relu',input_shape=(3,)))

# Hidden Layer 2
model.add(Dense(16, activation='sigmoid'))

# Output Layer
model.add(Dense(5, activation='softmax'))

# Compile Parameters
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# One-Hot Encode Labels
y_encoded = utils.to_categorical(y, num_classes=5)

# Train Model
model.fit(X, y_encoded, epochs=20, batch_size=32)

"""
Notes:
Input Layer: 3 Neurons
Hidden Layer 1: 32 Neurons Relu Activation
Hidden Layer 2: 50 Neurons Sigmoid Activation
Output Layer 1: 1 Neurons Softmax Activation
Compiler Loss: Categorical Cross-Entropy
"""

#-----------------------------------------------------------------------------#
#                          Neural Network: Regression                         #
#-----------------------------------------------------------------------------#


# create model
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(1))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X2,y2, epochs=25, batch_size=32)

"""
Notes:
Input Layer: 4 Neurons
Hidden Layer 1: 10 Neurons Relu Activation
Output Layer 1: 1 Neurons
Compiler Loss: MSE (Mean Squared Error)
"""












