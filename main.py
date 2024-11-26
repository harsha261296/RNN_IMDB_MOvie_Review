# Step1 include all the libraries 
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# load the IMDB dataset
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

##Load the H5 model
model=load_model('simple_RNN_IMDB1.h5')


#Step 2 : include the helper function

#decode the review function

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])


## FOr padding and preprocessing

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review



## step 3 is prediction model

def predict_sentiment(review):
    preprocessed_text=preprocess_text(review)
    prediction=model.predict(preprocessed_text)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]




## Create streamlit app

import streamlit as st

st.title("IMDB Movie review sentiment analysis")
st.write("Enter a movie review to classify it as positive or negative")

# user input

user_input=st.text_area("Movie Review")

if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)

    ## Make prediction
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0] >0.5 else 'Negative'

    st.write(f'The sentiment for the current review is : {sentiment}')
    st.write(f'The Score for the given review : {prediction[0][0]}')

else:
    st.write('Please enter a movie review')