import streamlit as st
import joblib 
import numpy as np

model = joblib.load('language_detection_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

confidence_threshold = 0.75

st.title("Kannada v/s Marathi Language Detector")
st.write("Enter text to detect if it's Kannada or Marathi:")
user_input = st.text_area("Input Text", "")
if user_input:
    #preprocess input text
    input_features = vectorizer.transform([user_input])

    probabilities = model.predict_proba(input_features)[0]
    predicted_class = model.classes_[np.argmax(probabilities)]
    confidence_score = np.max(probabilities)

    #Display results
    st.write("### Prediction Results:")
    if confidence_score < confidence_threshold:
        st.error("The input text is neither Kannada nor Marathi")
    else:
        st.success(f"Predicted Language: **{predicted_class}**")
        st.write(f"Confidence Score: **{confidence_score:.2f}**")

