import streamlit as st
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import re
import string
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


# Load saved model and vectorizer
model = pickle.load(open('best_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()

# Function to clean input text (same logic as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection App")
st.write("Check if a news article is **Real** or **Fake** using Machine Learning")

user_input = st.text_area("Enter news headline or article text here:")

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        if prediction == 1:
            st.success("✅ This news seems **REAL**.")
        else:
            st.error("❌ This news seems **FAKE**.")
