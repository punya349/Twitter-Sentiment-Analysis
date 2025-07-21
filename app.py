# install streamlit: pip install streamlit
# run: streamlit run app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load preprocessed data (optional, or use pre-fitted model)
@st.cache_resource
def load_model():
    df = pd.read_csv('twitter_sentiment.csv', header=None, index_col=[0])
    df = df[[2, 3]].reset_index(drop=True)
    df.columns = ['sentiment', 'text']

    # Vectorization
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['text'])
    y = df['sentiment']

    model = RandomForestClassifier()
    model.fit(X, y)

    return model, tfidf

model, tfidf = load_model()

# Streamlit UI
st.title("Twitter Sentiment Analyzer")
st.write("Enter a tweet below and click 'Analyze' to see its sentiment.")

user_input = st.text_area("Tweet Text", "")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        vectorized_input = tfidf.transform([user_input])
        prediction = model.predict(vectorized_input)[0]
        st.success(f"Predicted Sentiment: *{prediction}*")
