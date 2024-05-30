import streamlit as st
import time
from models.predict_toxicity import predict_toxicity

def toxicity_page():
    st.title("Оценка степени токсичности сообщения")
    message_input = st.text_area("Введите ваше сообщение:")

    if st.button("Оценить токсичность"):
        if message_input:
            start_time = time.time()
            prediction, probability = predict_toxicity(message_input)
            predict_time = time.time() - start_time
            st.write(f"Степень токсичности: {prediction} (вероятность: {probability:.4f}, время: {predict_time:.4f} секунд)")

toxicity_page()