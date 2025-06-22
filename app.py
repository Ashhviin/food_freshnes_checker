import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown
from PIL import Image

# Set page config
st.set_page_config(page_title="Food Freshness Checker", layout="centered")

# Function to load model (downloads if not found)
@st.cache_resource
def load_model():
    model_path = 'model/food_classifier.h5'
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive..."):
            os.makedirs('model', exist_ok=True)
            gdown.download(
                'https://drive.google.com/uc?id=1KazHcVqnjM2zm66fY9bbEcmVfop50aCI',
                model_path,
                quiet=False
            )
    return tf.keras.models.load_model(model_path)

# Load the model
model = load_model()

# Title
st.title("🥗 Food Freshness Classifier")
st.markdown("Upload an image of a vegetable or fruit, and the model will classify it as **Fresh**, **Stale**, or **Rotten**.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    class_names = ['Fresh', 'Rotten', 'Stale']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show result
    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**C**

    elif label == 'stale':
        st.warning("⚠️ Consider consuming soon or using in cooking.")
    else:
        st.error("❌ Spoiled – please compost or discard.")
