# following this article: https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
# :)
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils

with open("/Users/chris/github/py_text_generator/Sample_Texts/KJV_Revelations.txt") as text:
    text=text.read().lower()
    #print(text)

    # mapping characters to numbers in a dictionary so that they can be referenced faster later
    characters = sorted(list(set(text)))
    n_to_char = {n:char for n, char in enumerate(characters)}
    char_to_n = {char:n for n, char in enumerate(characters)}
    # X is the training array, Y is the target array
    # X will contain a set of characters and Y will contain the prediction for the following character(s)
    # example: X = [h,e,l,l] Y = [o]
    X = []
    Y = []

    length = len(text)
    # seq_length is how many characters will sit in X for Y to use for prediction 
    seq_length = 100
    # iterate over the text and create a bunch of training sequences
    for i in range(0,length-seq_length,1):
        sequence = text[i:i + seq_length]  
        label = text[i + seq_length]
        X.append([char_to_n[char] for char in sequence])
        Y.append(char_to_n[label])
    #reshaping and scaling X for better training, removing ordinal relationships from Y  
    X_modified = np.reshape(X, (len(X), seq_length, 1))
    X_modified = X_modified / float(len(characters))
    Y_modified = np_utils.to_categorical(Y)
    # building the model...
    model = Sequential()
    model.add(LSTM(400, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(400))
    model.add(Dropout(0.2))
    model.add(Dense(Y_modified.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(X_modified, Y_modified, epochs=1, batch_size=100)

    #model.save_weights('/Users/chris/github/py_text_generator/models/text_generator_400_0.2_400_0.2_baseline.h5')
    #model.load_weights('/Users/chris/github/py_text_generator/models/text_generator_400_0.2_400_0.2_baseline.h5')

    string_mapped = X[99]
    full_string = [n_to_char[value] for value in string_mapped]
    for i in range(seq_length):
        x = np.reshape(string_mapped,(1,len(string_mapped), 1))
        x = x / float(len(characters))
        pred_index = np.argmax(model.predict(x, verbose=0))
        seq = [n_to_char[value] for value in string_mapped]
        string_mapped.append(pred_index)
        string_mapped = string_mapped[1:len(string_mapped)]

    txt = ""
    for char in full_string:
        txt = txt+char
    print(txt)
