import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Tomato Disease Detector", layout="centered")

# --- LOAD MODEL ---
# Using @st.cache_resource ensures the model stays in memory and doesn't reload on every click
@st.cache_resource
def get_model():
    return load_model("tomato_model.keras")

model = get_model()

# --- LOAD CLASS NAMES ---
if os.path.exists("dataset"):
    class_names = sorted(os.listdir("dataset"))
else:
    # Fallback if the dataset folder isn't present on the server
    class_names = ["Bacterial Spot", "Early Blight", "Late Blight", "Healthy"]

# --- UI ELEMENTS ---
st.title("Disease Classifier")
st.write("Upload a photo of a tomato leaf to detect potential diseases.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # 2. Pre-processing
    # Convert PIL image to RGB and then to a NumPy array
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 3. Prediction
    with st.spinner('Analyzing...'):
        prediction = model.predict(img_array)[0]
        class_index = np.argmax(prediction)
        confidence = prediction[class_index]
        label = class_names[class_index]

    # 4. Show Results
    st.divider()
    st.subheader(f"Prediction: **{label}**")
    st.progress(float(confidence))
    st.write(f"**Confidence Score:** {confidence:.2%}")

    # Optional: Show breakdown
    with st.expander("See probability breakdown"):
        for name, prob in zip(class_names, prediction):
            st.write(f"{name}: {prob:.2%}")