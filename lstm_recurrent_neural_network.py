"""
Title: Recurrent Neural Network: Classification
Author: Rodrigo Rangel
Description: Recurrent Neural Networks (RNNs) differ slightly from traditional
             networks. Below we will cover the functionality of RNNs and differences
             
			 * The output for the current state will be an input for the next
               state
             * At each element in the sequence, the learners consider not only the
               current input, but also the previous elements ('remember' functionality)
             * Learns long-term dependencies in a sequence, for example:
                 a. Next Word Predictions
                 b. Sentiment Classification
             * Designed to work with sequential data such as:
			     a. Text
				 b. Timeseries
             
             In this examples, we will use imdb dataset to train an LSTM in order to make 
             sentiment predictions on reviews and make predictions on a live review
"""

#-----------------------------------------------------------------------------#
#                                Libraries                                    #
#-----------------------------------------------------------------------------#

# Simple LSTM for sequence classification
import numpy
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)

#-----------------------------------------------------------------------------#
#                                imdb.load_data                               #
#-----------------------------------------------------------------------------#

"""
Note:
y_train/y_test:
    0: Negative
    1: Positive

X_train, X_test:
    0: Padding (at beginiing)
    1: Start of text
    2: Unknown (If word < top words)
    3: Unused
"""

top_words = 5000 # Use only top 5000 words
start_index_from = 3 # Word index offset

# load the dataset but only keep the top n words, zero the rest
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words, index_from = start_index_from)

# Convert from numeric back to string
word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+start_index_from) for k,v in word_to_id.items()}

word_to_id["<PAD>"] = 0    # Pad at beginning
word_to_id["<START>"] = 1  # Start
word_to_id["<UNK>"] = 2    # Unknown
word_to_id["<UNUSED>"] = 3 # Unused

id_to_word = {value:key for key,value in word_to_id.items()}
print(' '.join(id_to_word[ide] for ide in X_train[0]))

#-----------------------------------------------------------------------------#
#                              Building LSTM                                  #
#-----------------------------------------------------------------------------#

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# LSTM model
embedding_vecor_length = 32
model = Sequential()

# Embedding Layer: Used to create word vectors for incoming words
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#-----------------------------------------------------------------------------#
#                            Testing On Sentence                              #
#-----------------------------------------------------------------------------#

def convert_class(prob):
    if prob > .50:
        return 'Positive'
    else:
        return 'Negative'

# Random Test Case
test_sentence = 'This movie was very good!'.lower() # Rigo # Positive

# Pre-processing test-sentence
test_sentence = test_sentence.split(' ')
test_sentence.insert(0, '<START>')

# Convert to numeric (mapping)
encoded_test_sentence = []
for temp_id in test_sentence:
    try:
        encode = word_to_id[temp_id]
        if encode > 5000:
            encode = 2
        encoded_test_sentence.append(encode)
    except:
        encoded_test_sentence.append(2)

# Pad
padded_encoded_test_sentence = sequence.pad_sequences([encoded_test_sentence], maxlen=max_review_length)


predicted_prob = model.predict(padded_encoded_test_sentence)[0][0]

print(convert_class(predicted_prob))
