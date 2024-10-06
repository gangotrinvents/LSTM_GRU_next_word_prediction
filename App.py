# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

import streamlit as st

with( open('data_tokenizer.pkl', 'rb') ) as file:
    data_tokenizer = pickle.load(file)

model = load_model('LSTM_predict.h5')

def lstm_prediction(inp):
    val_token = data_tokenizer.texts_to_sequences([inp])[0]
    pad_seq = pad_sequences([val_token], padding='pre', maxlen= 14)
    pred = model.predict(pad_seq)
    for word, index in data_tokenizer.word_index.items():
        if index == np.argmax(pred):
            return word
    return None

st.title('Next word prediction')
inp = st.text_input('Please enter the string for which you wanted to predict next word')

if st.button('Generate'):
    st.write('Next word: ', lstm_prediction(inp))