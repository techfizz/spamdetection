import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load and preprocess data
data = pd.read_csv("C:\\Users\\user\\Desktop\\spamdetection\\spam.csv")
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

mess = data['Message']
cat = data['Category']
mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2)

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

# Train model
model = MultinomialNB()
model.fit(features, cat_train)

# Prediction function
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result[0]

# Streamlit Web App Design
st.markdown(
    """
    <style>
    body {
        background-color: #e2d1f9;
        font-family: Arial, sans-serif;
    }
    .main-header {
        color: white;
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 20px;
        padding: 20px;
        background: linear-gradient(90deg, rgb(37, 22, 243), #8bc34a);
        border-radius: 10px;
    }
    .subtext {
        color: #555;
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .result-box {
        font-size: 24px;
        text-align: center;
        font-weight: bold;
        margin-top: 30px;
        padding: 20px;
        border-radius: 10px;
    }
    .spam {
        background-color: #ff6b6b;
        color: white;
    }
    .not-spam {
        background-color: #4caf50;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-header">🚀 Spam Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Detect whether your message is spam or not with ease!</div>', unsafe_allow_html=True)

# Input Section
input_message = st.text_input("📩 Enter your message below:")
if st.button("🔍 Check Message"):
    if input_message.strip():
        # Perform prediction
        result = predict(input_message)
        result_class = "Spam" if result == "Spam" else "Not Spam"
        result_style = "spam" if result == "Spam" else "not-spam"
        
        # Display result dynamically
        st.markdown(
            f'<div class="result-box {result_style}">{result_class}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("⚠️ Please enter a valid message!")

# Footer
st.markdown(
    """
    <hr>
    <div style="text-align: center; font-size: 14px; color: #888;">
    Made with ❤️ using Streamlit | <a href="https://streamlit.io/" target="_blank" style="color: #4CAF50; text-decoration: none;">Learn More</a>
    </div>
    """,
    unsafe_allow_html=True,
)
