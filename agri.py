import streamlit as st
from streamlit_option_menu import option_menu
import requests
import json
import pandas as pd
import numpy as np
import pickle
from translate import Translator
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Configuration for the Chatbot
OLLAMA_URL = "http://localhost:11434/api/generate"

# Translation function
def translate_text(text, lang='en'):
    translator = Translator(to_lang=lang)
    return translator.translate(text)

# Recommendation function
def generate_recommendations(N, P, K, temp, humidity, ph, rainfall, pest_status, crop):
    recs = [f"Recommended Crop: {crop}"]
    if N < 50:
        recs.append("Add nitrogen-rich fertilizer.")
    if P < 40:
        recs.append("Add phosphorus-rich fertilizer.")
    if K < 40:
        recs.append("Add potassium-rich fertilizer.")
    if humidity < 60:
        recs.append("Irrigate: Humidity is low.")
    if rainfall < 100:
        recs.append("Consider additional irrigation.")
    if ph < 6.0:
        recs.append("Add lime to increase soil pH.")
    if pest_status == "Pest Detected":
        recs.append("Apply organic pesticide.")
    return recs

# Train and save the crop prediction model (if not already saved)
try:
    with open('crop_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    data = pd.read_csv('Crop_recommendation.csv')
    X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = data['label']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    with open('crop_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Page 1: Home Page with Crop Recommendation Form
def page1():
    st.title("Agricultural Bot Assistant")
    st.write("""
        Welcome to the Agricultural Bot! This bot assists small-scale farmers with:
        - **Soil Testing**: Real-time soil sensor data.
        - **Pest Detection**: Image recognition for pest identification.
        - **Crop Management**: Irrigation, fertilization, and harvesting advice.
        - **Financial Aids**: Information on local government subsidies.
        - **Multi-language Support**: Available in multiple languages.
        - **Engaging Animations**: Interactive visuals to enhance user experience.
    """)

    st.subheader("Crop Recommendation Form")
    with st.form(key="crop_form"):
        N = st.number_input("Nitrogen (N)", min_value=0, step=1, value=0)
        P = st.number_input("Phosphorus (P)", min_value=0, step=1, value=0)
        K = st.number_input("Potassium (K)", min_value=0, step=1, value=0)
        temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1, value=20.0)
        humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, step=1, value=50)
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1, value=6.5)
        rainfall = st.number_input("Rainfall (mm)", min_value=0, step=1, value=100)
        image = st.file_uploader("Upload Crop Image (Optional for Pest Detection)", type=["jpg", "jpeg", "png"])
        language = st.selectbox("Language", options=["en", "es", "ta"], format_func=lambda x: {"en": "English", "es": "Spanish", "ta": "Tamil"}[x])
        submit_button = st.form_submit_button("Get Recommendations")

    if submit_button:
        input_data = pd.DataFrame([[N, P, K, temp, humidity, ph, rainfall]], 
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        predicted_crop = model.predict(input_data)[0]
        pest_status = "No Pest" if image is None else "Pest Detected"
        recommendations = generate_recommendations(N, P, K, temp, humidity, ph, rainfall, pest_status, predicted_crop)
        translated_recs = [translate_text(rec, lang=language) for rec in recommendations]
        st.subheader("Results")
        for rec in translated_recs:
            st.write(rec)

# Page 2: Crop Chatbot (Unchanged)
def page2():
    st.title("Crop Chatbot")
    st.write("Ask me anything about crops, soil, pests, or farming!")

    user_input = st.text_input("You: ", key="chat_input")
    if st.button("Send"):
        if user_input.lower() in ["exit", "quit"]:
            st.write("AI: Goodbye!")
        else:
            response = requests.post(OLLAMA_URL, json={"model": "mistral", "prompt": user_input, "stream": False})
            try:
                reply = response.json().get("response", "Error: No response")
                st.write("AI:", reply)
            except json.JSONDecodeError:
                st.write(f"AI: Error: {response.text}")

# Page 3: Plot Page with Five Plots
def page3():
    st.title("Farm Plot Visualization")
    st.write("Explore interactive visualizations of your farm data.")

    # Generate sample data
    x = np.linspace(0, 10, 100)
    soil_health = np.sin(x) + np.random.normal(0, 0.1, 100)
    crop_yield = np.cos(x) + np.random.normal(0, 0.1, 100)
    temp_trend = 20 + 5 * np.sin(x / 2)
    humidity_level = 50 + 20 * np.cos(x / 3)
    rainfall_pattern = 100 + 30 * np.sin(x / 4)

    # Plot 1: Soil Health
    fig1 = go.Figure(data=go.Scatter(x=x, y=soil_health, mode='lines+markers', name='Soil Health'))
    fig1.update_layout(title='Soil Health Over Time', xaxis_title='Time', yaxis_title='Health Index')
    st.plotly_chart(fig1)

    # Plot 2: Crop Yield
    fig2 = go.Figure(data=go.Scatter(x=x, y=crop_yield, mode='lines+markers', name='Crop Yield'))
    fig2.update_layout(title='Crop Yield Over Time', xaxis_title='Time', yaxis_title='Yield (tons)')
    st.plotly_chart(fig2)

    # Plot 3: Temperature Trend
    fig3 = go.Figure(data=go.Scatter(x=x, y=temp_trend, mode='lines+markers', name='Temperature'))
    fig3.update_layout(title='Temperature Trend', xaxis_title='Time', yaxis_title='Temperature (°C)')
    st.plotly_chart(fig3)

    # Plot 4: Humidity Level
    fig4 = go.Figure(data=go.Scatter(x=x, y=humidity_level, mode='lines+markers', name='Humidity'))
    fig4.update_layout(title='Humidity Levels', xaxis_title='Time', yaxis_title='Humidity (%)')
    st.plotly_chart(fig4)

    # Plot 5: Rainfall Pattern
    fig5 = go.Figure(data=go.Scatter(x=x, y=rainfall_pattern, mode='lines+markers', name='Rainfall'))
    fig5.update_layout(title='Rainfall Pattern', xaxis_title='Time', yaxis_title='Rainfall (mm)')
    st.plotly_chart(fig5)

# Main App
st.set_page_config(page_title="Agricultural Bot", layout="wide")

# Navigation
with st.sidebar:
    selected = option_menu("Menu", ["Home", "Chatbot", "Plots"], 
                          icons=['house', 'chat', 'bar-chart'], menu_icon="cast", default_index=0)

# Display selected page
if selected == "Home":
    page1()
elif selected == "Chatbot":
    page2()
elif selected == "Plots":
    page3()

    