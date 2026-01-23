import streamlit as st
from inference import predict_text
st.set_page_config(
  page_title='Mental Health NLP Analyzer',
  layout='centered'
)
st.title("Mental Health Text Analyzer")
st.write(
   "This application analyzes text using a trained NLP model "
    "to identify potential mental health signals."
)
st.warning(
    "⚠️ This tool is for academic and research purposes only. "
    "It is not a medical diagnostic system."
)
user_text = st.text_area(
  "Enter text to analyze:",
  height = 180,
  placeholder='Type or paste text here...'
)

if st.button("Analyze"):
  if user_text.strip()=='':
    st.error('Please enter some text.')
  else:
    label,probs,explanation =predict_text(user_text)
    st.subheader("Prediction")
    st.success(label)

    st.subheader("Confidence Scores")
    for cls, score in probs.items():
      st.write(f"{cls}: {round(score * 100,2)}%")

    if explanation:
      st.subheader("Explanation")
      st.info(explanation)
  

















