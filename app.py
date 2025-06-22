import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/food_classifier.h5')

model = load_model()
class_names = ['fresh', 'stale', 'spoiled']

def predict(img: Image.Image):
    img = img.resize((224,224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    return class_names[idx], float(preds[idx])

st.title("🍏 Food Freshness Checker")
uploaded = st.file_uploader("Upload a food image...", type=["jpg","png"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)
    label, conf = predict(img)
    st.success(f"Prediction: **{label.upper()}** (confidence: {conf:.1%})")
    if label == 'fresh':
        st.info("👍 You can safely consume it.")
    elif label == 'stale':
        st.warning("⚠️ Consider consuming soon or using in cooking.")
    else:
        st.error("❌ Spoiled – please compost or discard.")
