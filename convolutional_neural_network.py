"""
Title: Convolutional Neural Network: Architecture
Author: Rodrigo Rangel
Description: * This script focuses on fully understanding the architecture
               behind Convolutional Neural Networks (CNN)	  
             * CNNs are heavily used in computer vision, image recognition, 
               and object detection
             * The architecture of a CNN is as follows:
                 1. Input Layer
                 2. Convolutional Layer
                 3. Pooling Layer
                 4. Fully Connected Layer
                 5. Output Layer

             * A deep CNN will consist of multiple Convolutional and Pooling
               layers (steps 3 and 4 sequentially)

"""  
#-----------------------------------------------------------------------------#
#                              Dependencies                                   #
#-----------------------------------------------------------------------------#

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from IPython.display import Image

#-----------------------------------------------------------------------------#
#                          CNN: Initializing Model                            #
#-----------------------------------------------------------------------------#

# Creating Deep Neural Network (DNN) object
model = Sequential()

#-----------------------------------------------------------------------------#
#                          CNN: Conv Layer 1                                  #
#-----------------------------------------------------------------------------#

"""
Notes:
2D Convulutional Layer
Number of Filters: 32
Kernel Size of Sliding Window: 5x5
Kernel Size of Strides: 1x1
Activation Function: ReLu (Rectified Linear Units)
Input Shape: 28x28x1 (Image Dimension)
"""

model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(28,28,1)))

"""
No weights or bias variables need to be declared when using Keras
In order to avoid dimensionality reduction on this step, the image is padded
"""

#-----------------------------------------------------------------------------#
#                          CNN: Pool Layer 1                                  #
#-----------------------------------------------------------------------------#

# Pool Layer 1
"""
Pool Window Size: 2x2
Pool Window Stride: 2x2
"""
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#-----------------------------------------------------------------------------#
#                          CNN: Conv Layer 2                                  #
#-----------------------------------------------------------------------------#

"""
2D Convulutional Layer
Number of Filters: 64
Kernel Size of Sliding Window: 5x5
Kernel Size of Strides: 1x1 (Default)
Activation Function: ReLu (Rectified Linear Units)
Input Shape: Defined by previous layer (28,28,32)
"""

model.add(Conv2D(64, (5, 5), activation='relu'))

#-----------------------------------------------------------------------------#
#                          CNN: Pool Layer 2                                  #
#-----------------------------------------------------------------------------#

# Pool Layer 2
"""
Pool Window Size: 2x2
Pool Window Stride: Default
"""
model.add(MaxPooling2D(pool_size=(2, 2)))

#-----------------------------------------------------------------------------#
#                          CNN: Fully Connected Layer                         #
#-----------------------------------------------------------------------------#

# Fully Connected
"""
Flatten: Final Pool Layer (7x7x64) (vectorize)
Dense: Layer 1000 neurons
Dense: Final Output 10 Classes
"""
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))

#-----------------------------------------------------------------------------#
#                     CNN: Compiling, Training, & Evaluating                  #
#-----------------------------------------------------------------------------#

# Compiling 
"""
Loss Function: Cross Entropy
Optimizer: Adam
Learning Rate: .01
Metric: Accuracy
"""
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

# Fit Model
batch_size = 128
epochs = 10

model.fit(1, 1,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1, # Display progress to console
          validation_data=(1, 1)
          #callbacks=[history]
         )
		 
# Model Score
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])








