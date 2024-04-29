import streamlit as st
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Download NLTK data
import nltk
nltk.download('stopwords')

# Function for stemming
def stemming(content):
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Function to predict sentiment
def predict_sentiment(tweet, model, vectorizer):
    stemmed_tweet = stemming(tweet)
    vectorized_tweet = vectorizer.transform([stemmed_tweet])
    prediction = model.predict(vectorized_tweet)
    return prediction[0]

# Load the trained model and vectorizer
def load_model():
    model = pickle.load(open('trained_model.sav', 'rb'))
    vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
    return model, vectorizer

# Streamlit app
def main():
    st.title('Twitter Sentiment Analysis')
    st.markdown('## Enter your tweet below:')
    tweet_input = st.text_input('')

    if st.button('Predict'):
        if tweet_input:
            model, vectorizer = load_model()
            prediction = predict_sentiment(tweet_input, model, vectorizer)
            if prediction == 0:
                st.write("It is a negative tweet.")
            else:
                st.write("It is a positive tweet.")
        else:
            st.warning("Please enter a tweet.")

if __name__ == '__main__':
    main()
