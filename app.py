from flask import Flask, request, render_template 
import re
import numpy as np
import joblib
from gensim.downloader import load
from sklearn.base import BaseEstimator, TransformerMixin

# # Define the GloVeVectorizer class
class GloVeVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y=None):
        # No fitting necessary for pre-trained embeddings
        return self

    def transform(self, X):
        return np.vstack([self.document_vector(doc) for doc in X])

    def document_vector(self, doc):
        """Remove out-of-vocabulary words. Create document vectors by averaging word vectors."""
        # Filter out-of-vocabulary words
        vocab_tokens = [word for word in doc if word in self.model]

        if not vocab_tokens:
            # If there are no tokens in the vocabulary, return a zero vector
            return np.zeros(self.model.vector_size)

        # Compute the mean vector of the tokens
        return np.mean(self.model[vocab_tokens], axis=0)

app = Flask(__name__)

# Load the saved model
model = joblib.load('best_models/demo_model_rfc_hpy.pkl')

# Load the pre-trained GloVe embeddings
wv = load('glove-twitter-50')

# Define preprocess_text and classify_sentiment functions as before...

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Function to classify sentiment
def classify_sentiment(text):
    # Preprocess the text
    text = preprocess_text(text)
    
    # Instantiate GloVeVectorizer
    glove_vectorizer = GloVeVectorizer(model=wv)

    # Transform text into document vector using GloVe embeddings
    text_embedding = glove_vectorizer.transform([text])

    # Predict sentiment
    prediction = model.predict(text_embedding)
    
    # Map predictions to sentiment labels
    if prediction[0] <= 2:
        return "Negative"
    elif prediction[0] == 3:
        return "Neutral"
    else:
        return "Positive"
    # return prediction[0]

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def prediction():
    if request.method == 'POST':
        review = request.form.get("test_string")
        sentiment = classify_sentiment(review)
        return render_template("result.html", review=review, prediction=sentiment)
    return render_template("index.html")  # Render the form if the method is GET

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
