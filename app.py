import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import json
import requests
import zipfile
import os

# --- SETUP PAGE ---
st.set_page_config(page_title="Fish Freshness Detector", page_icon="🐟", layout="centered")

# --- FUNCTION TO LOAD LOTTIE ANIMATIONS ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- LOAD LOTTIE ANIMATIONS ---
lottie_fresh = load_lottieurl("https://lottie.host/f9acdb34-180b-41a6-8670-f579ae290779/V0fhWcwSKt.json")
lottie_moderate = load_lottieurl("https://lottie.host/158eabfc-4d8d-4964-a2d6-f61a4d7d86f7/XGVqULGrWu.json")
lottie_spoiled = load_lottieurl("https://lottie.host/49b79f94-2c7e-4685-a96c-5c999f351f2b/xkRaWhpAkz.json")

# --- LOAD MODEL FROM ZIP ---
@st.cache_resource
def load_freshness_model():
    zip_path = "model.zip"  # ✅ Corrected path
    extract_dir = "temp_model_folder"
    model_filename = "model.tflite"

    try:
        # Extract model only once
        if not os.path.exists(os.path.join(extract_dir, model_filename)):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

        model_path = os.path.join(extract_dir, model_filename)
        return load_model(model_path)

    except Exception as e:
        st.error(f"❌ Error loading model from ZIP: {e}")
        return None

model = load_freshness_model()

# --- CLASS DATA ---
class_names = ['Fresh', 'Moderately Fresh', 'Spoiled']
lottie_map = {
    'Fresh': lottie_fresh,
    'Moderately Fresh': lottie_moderate,
    'Spoiled': lottie_spoiled
}
custom_messages = {
    'Fresh': (
        "**✅ Fresh Fish Detected**\n\n"
        "- Estimated Age: Less than 1 day old\n"
        "- Features: Bright eyes, red gills, firm flesh\n"
        "- Suitable for raw and cooked dishes."
    ),
    'Moderately Fresh': (
        "**⚠️ Moderately Fresh Fish Detected**\n\n"
        "- Estimated Age: 2–3 days old\n"
        "- Slightly dull eyes and minor odor\n"
        "- Cook thoroughly before consuming."
    ),
    'Spoiled': (
        "**🚫 Spoiled Fish Detected**\n\n"
        "- Estimated Age: 4–5+ days old\n"
        "- May contain formalin or show signs of decay\n"
        "- Unsafe for consumption."
    )
}

# --- SIDEBAR ---
st.sidebar.image("https://img.icons8.com/color/96/fish-food.png", width=100)
st.sidebar.title("About")
st.sidebar.info("""
This app predicts fish freshness using AI.  
Upload a clear photo of a fish 🐟 and get an instant analysis!

**Built by [jaydish kennedy.j]**
""")

# --- MAIN TITLE ---
st.title("🐟 AI-Based Fish Freshness Detector")
st.markdown("Upload a fish image to analyze its freshness using a deep learning model.")

# --- IMAGE UPLOAD ---
uploaded_file = st.file_uploader("📤 Upload a fish image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Uploaded Image", use_column_width=True)

    # --- PREPROCESS IMAGE ---
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array / 255.0, axis=0)

    # --- PREDICT ---
    if model:
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        predicted_class = class_names[predicted_index]

        # --- RESULT SECTION ---
        st.markdown(f"## 🎯 Prediction: **{predicted_class}**")
        st.markdown(f"**Confidence:** `{confidence*100:.2f}%`")
        st.markdown("📢 " + custom_messages[predicted_class])

        # --- DISPLAY LOTTIE ANIMATION ---
        st.components.v1.html(
            f"""
            <lottie-player src="{requests.get(lottie_map[predicted_class]['v']).url}" 
                           background="transparent" speed="1" 
                           style="width: 300px; height: 300px;" loop autoplay>
            </lottie-player>
            """, height=300
        )

        # --- SHOW CLASS PROBABILITIES ---
        st.subheader("📊 Class Probabilities")
        for i, class_name in enumerate(class_names):
            st.progress(prediction[0][i])
            st.text(f"{class_name}: {prediction[0][i]*100:.2f}%")

        # --- DOWNLOADABLE REPORT ---
        report_text = f"""Fish Freshness Prediction Report

Prediction: {predicted_class}
Confidence: {confidence*100:.2f}%

{custom_messages[predicted_class]}
"""
        st.download_button("📥 Download Report", report_text, file_name="FishFreshnessReport.txt")
