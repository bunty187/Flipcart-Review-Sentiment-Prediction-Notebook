# Flipcart-Review-Sentiment-Prediction-Notebook

## Objective
The Objective of this project is to classify customer reviews as positive or negative and understand the pain points of customers who write negative reviews. By analyzing the sentiment of reviews, we aim to gain insights into product features that contribute to customer satisfaction or dissatisfaction.

## Dataset
The dataset consists of 8518 reviews for the "YONEX_MAVIS_350_NYLON_SHUTTLE" product from Flipcart. Each Review includes features such as Reviewer Name, Rating, Review Text, Place of Review, Date of Review, Up Votes, and Down Votes.

### Data Preprocessing
1. Text Cleaning: Remove Special Characters, Punctuation, and stopwords from the review text.
2. Text Normalization: Perform Lemmatization or stemming to reduce words to their base forms.
3. Numerical Feature Extraction: Apply techniques like Bag-of-words (BoW), Term Frequency-Inverse Document Frequency(TF-IDF), Word2Vec(W2V), and BERT models for feature extraction.

### Modeling Approach
1. Model Selection: Train and evaluate various machine learning and deep learning models using the embedded text data.
2. Evaluation Metric:  Use the F-1 Score as the evaluation metric to assess the performance of the models in classify sentiment.

### Model Deployment
1. Flask or Streamlit App Development: Develop a Flask or Sentiment web application that takes user input in the form of a review and generates the sentiment (Positive or Negative) of the review.
2. Model Integration: Integrate the trained sentiment classification model into the Flask app for real-time inference.
3. Deployment: Deploy the Flask app on an AWS EC2 instance to make it accessible over the internet.

### Workflow
1. Data Loading and Analysis: Gain Insights into product features that contribute to customer satisfaction or dissatisfaction.
2. Data Cleaning: Preprocess the review text by removing noise and normalizing the text.
3. Text Embedding: Experiment with different text embedding techiques to represent the review text as numerical vectors.
4. Model Training: Train Machine Learning and Deep learning model on the embedded text data to classify sentiment while logging relevant information with MLFlow.
5. Model Evaluation: Evaluate the performance of the trained models using the F1-Score metric. .
6. Flask App Development: Develop a Flask web application for sentiment analysis of user-provided reviews.
7. Model Deployment: Deploy the trained sentiment classification model along with Flask app on an AWS EC2 instance.
8. Testing and Monitoring: Test the deployed application and monitor its performance for any issues or errors. Demonstrate how to log parameters, metrics, and artifacts using MLFlow tracking APIs. Customizing MLFlow UI with run names.Demonstrate metric plots. Demonstrate hyperparameters plots. Demonstrate how to register models and manage by tagging them.
9. Build a Prefect workflow and Auto Schedule it. Show the Prefect Dashboard with relevant outputs.
