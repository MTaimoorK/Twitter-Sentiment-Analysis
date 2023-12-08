# _Twitter Sentiment Analysis using ML_

## Author: Muhammad Taimoor Khan

## Overview
This project focuses on sentiment analysis of Twitter data using Machine Learning techniques. The goal is to classify tweets into either positive or negative sentiment.

## Project Structure
The project is structured into several sections, including data preprocessing, stemming, vectorization, model training, evaluation, and model deployment.

## Dependencies
Make sure to install the required dependencies before running the code. You can install them using the following commands:
```bash
pip install pandas numpy scikit-learn nltk
```

## Dataset
The dataset used for training and testing the model is stored in the ./data directory. It is a CSV file with columns such as 'target', 'id', 'date', 'flag', 'user', and 'text'.

## Data Preprocessing
The dataset undergoes various preprocessing steps, including handling missing values, renaming columns, and converting labels. The 'target' column is modified to have two labels: 0 for negative tweets and 1 for positive tweets.

## Stemming
Stemming is applied to reduce words to their root form. The NLTK library is used for this purpose, and the processed data is stored in the 'stemmed_content' column.

## Vectorization
Textual data is converted into numerical form using the Term Frequency-Inverse Document Frequency (TF-IDF) vectorization technique.

## Model Training
A Logistic Regression model is trained on the preprocessed and vectorized data.

## Model Evaluation
The accuracy of the model is evaluated on both training and testing data. The model achieved an accuracy of 81% on the training data and 77% on the testing data.

## Saving The Trained Model
The trained model is saved using the pickle library and stored as 'trained_model.sav'.

## Using The Saved Model
You can load the saved model and make predictions on new data. An example is provided in the code to demonstrate how to use the saved model for predictions.

## Files
- ```Twitter_Sentiment_Analysis.ipynb:``` Jupyter Notebook containing the entire code.
- ```trained_model.sav:``` Saved ML model for future use.
- ```./data/training.1600000.processed.noemoticon.csv:``` Original dataset file.

## Usage
Follow the steps outlined in the Jupyter Notebook to run the project. Ensure the required dependencies are installed.

## Future Improvements
- Fine-tuning hyperparameters for better model performance.
- Exploring and experimenting with other ML algorithms.
- Handling additional features for improved sentiment analysis.

## Contributions
Feel free to contribute and provide feedback!
