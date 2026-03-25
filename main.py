import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
import os
import shutil

from nltk.stem.porter import PorterStemmer

nltk_path = os.path.join(os.getcwd(), 'nltk_data')
os.environ["NLTK_DATA"] = nltk_path
nltk.data.path.append(nltk_path)

def force_download_nltk():
    # Remove corrupted punkt if any
    punkt_dir = os.path.join(nltk_path, 'tokenizers', 'punkt')
    if os.path.exists(punkt_dir):
        shutil.rmtree(punkt_dir, ignore_errors=True)

    nltk.download('punkt', download_dir=nltk_path, force=True)
    nltk.download('punkt_tab', download_dir=nltk_path, force=True)
    nltk.download('stopwords', download_dir=nltk_path, force=True)

force_download_nltk()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))


    return " ".join(y)

tfidf =pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Sms Spam Classification")

input_sms=st.text_area("Enter your message")

if st.button("Predict"):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)

    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
