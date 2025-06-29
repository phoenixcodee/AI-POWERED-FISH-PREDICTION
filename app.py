import streamlit as st
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import requests
import json
import streamlit.components.v1 as components

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

# --- LOAD TFLITE MODEL ---
@st.cache_resource
def load_freshness_model():
    try:
        interpreter = tflite.Interpreter(model_path="model_float16.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"❌ Error loading TFLite model: {e}")
        return None

interpreter = load_freshness_model()

# --- CLASS INFO ---
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

**Built with ❤️ by [jaydish kennedy.j]**
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
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    input_data = np.expand_dims(img_array, axis=0)

    # --- PREDICT USING TFLITE ---
    if interpreter:
        input_index = interpreter.get_input_details()[0]['index']
        output_index = interpreter.get_output_details()[0]['index']

        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_index)

        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        predicted_class = class_names[predicted_index]

        # --- RESULT SECTION ---
        st.markdown(f"## 🎯 Prediction: **{predicted_class}**")
        st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")
        st.markdown("📢 " + custom_messages[predicted_class])

        # --- DISPLAY LOTTIE ANIMATION ---
        if lottie_map[predicted_class]:
            components.html(
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
            progress_value = float(np.clip(prediction[0][i], 0.0, 1.0))  # ✅ Ensure safe float
            st.progress(progress_value)
            st.text(f"{class_name}: {progress_value * 100:.2f}%")

        # --- DOWNLOADABLE REPORT ---
        report_text = f"""Fish Freshness Prediction Report

Prediction: {predicted_class}
Confidence: {confidence * 100:.2f}%

{custom_messages[predicted_class]}
"""
        st.download_button("📥 Download Report", report_text, file_name="FishFreshnessReport.txt")
