import streamlit as st
import json
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Skin Disease Classification", layout="wide")

@st.cache_resource
def load_trained_model():
    return load_model('model.h5', compile=True)

model = load_trained_model()

f = open('dat.json')
data = json.load(f)
keys = list(data)

def Predict(image):
    img = np.array(image)
    img = cv2.resize(img, (32, 32)) / 255.0
    prediction = model.predict(img.reshape(1, 32, 32, 3))

    max_index = prediction.argmax()
    confidence = float(prediction[0][max_index]) * 100  

    return (keys[max_index], confidence,
            data[keys[max_index]]['description'], 
            data[keys[max_index]]['symptoms'], 
            data[keys[max_index]]['causes'], 
            data[keys[max_index]]['treatement-1'])

st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #D2A679;
        color: #ffffff;
    }
    .stTextInput, .stTextArea {
        background-color: #E0C1A3 !important;
        color: black !important;
    }
    .stProgress > div > div {
        background-color: #794D30 !important;
    }
    .stHeader {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="stHeader">Skin Disease Classification</h1>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    if uploaded_file is not None:
        name, confidence, description, symptoms, causes, treatment = Predict(image)

        st.text_input('Name Of Disease', value=name)
        st.progress(int(confidence))  
        st.text(f"Accuracy: {confidence:.2f}%")
        st.text_area('Description', value=description, height=100)
        st.text_area('Symptoms', value=symptoms, height=100)
        st.text_area('Causes', value=causes, height=100)
        st.text_area('Treatment', value=treatment, height=100)
