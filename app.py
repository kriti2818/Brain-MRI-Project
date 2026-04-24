import streamlit as st
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# -------------------- LOAD BG IMAGE --------------------
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_img = get_base64("background.jpeg")

# -------------------- CSS --------------------
st.markdown(f"""
<style>

/* Background Image */
.stApp {{
    background: url("data:image/jpg;base64,{bg_img}") no-repeat center center fixed;
    background-size: cover;
}}

/* Dark overlay */
.stApp::before {{
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(0,0,0,0.6);
    z-index: -1;
}}

/* Glass Card */
.container {{
    max-width: 520px;
    margin: auto;
    margin-top: 80px;
    padding: 35px;
    border-radius: 18px;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    box-shadow: 0 0 40px rgba(0,0,0,0.5);
    text-align: center;
}}

/* Title */
.title {{
    font-size: 34px;
    font-weight: 800;
    color: white;
}}

/* Subtitle */
.subtitle {{
    font-size: 15px;
    color: #e2e8f0;
    margin-bottom: 25px;
}}

/* Upload Box */
[data-testid="stFileUploader"] {{
    border: 2px dashed rgba(255,255,255,0.3);
    padding: 15px;
    border-radius: 12px;
}}

/* Button */
.stButton>button {{
    width: 100%;
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    color: white;
    font-size: 16px;
    font-weight: bold;
    border-radius: 10px;
    padding: 12px;
    border: none;
}}

.stButton>button:hover {{
    opacity: 0.9;
}}

/* Result box */
.result {{
    margin-top: 20px;
    padding: 15px;
    border-radius: 12px;
    background: rgba(255,255,255,0.1);
    font-size: 18px;
    color: white;
}}

</style>
""", unsafe_allow_html=True)

# -------------------- MODEL --------------------
model = load_model("brain_tumor_model.h5")
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# -------------------- UI --------------------
st.markdown('<div class="container">', unsafe_allow_html=True)

st.markdown('<div class="title">🧠 Brain Tumor Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload MRI image to detect tumor type</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, use_container_width=True)

    if st.button("Predict"):
        img_resized = cv2.resize(img, (224, 224))
        img_resized = img_resized / 255.0
        img_resized = np.reshape(img_resized, (1, 224, 224, 3))

        prediction = model.predict(img_resized)
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction)

        if predicted_class == "notumor":
            st.markdown(f'<div class="result">✅ No Tumor Detected<br>Confidence: {confidence:.2f}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result">⚠️ {predicted_class} detected<br>Confidence: {confidence:.2f}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)