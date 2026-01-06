import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

st.set_page_config(
    page_title="Email Spam Detector",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z]", " ", text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

st.title("📧 Email Spam Detection")
st.write("Classify emails using Machine Learning and NLP.")

email_text = st.text_area(
    "Enter Email Content",
    height=200,
    placeholder="Paste email text here..."
)

if st.button("Detect Spam"):
    if email_text.strip() == "":
        st.warning("Please enter email text.")
    else:
        cleaned = preprocess(email_text)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("🚨 Spam Email Detected")
        else:
            st.success("✅ This is NOT Spam")

with st.expander("Model Details"):
    st.markdown("""
    - **Model:** Linear Support Vector Machine  
    - **Features:** TF-IDF (Unigram + Bigram)  
    - **Preprocessing:** Stopword removal, stemming  
    - **Use Case:** Email security, phishing detection
    """)
