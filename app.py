import streamlit as st
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import requests
import json
import streamlit.components.v1 as components

# --- CONFIG ---
st.set_page_config(page_title="FishSnap App", page_icon="🐟", layout="centered")

# --- CUSTOM BACKGROUND STYLE ---
st.markdown(
    """
    <style>
    .stApp {
        background-image:url("fish.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 0 15px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- LOAD LOTTIE ---
@st.cache_resource
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model_float16.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# --- CLASS INFO ---
class_names = ['Fresh', 'Moderately Fresh', 'Spoiled']
lottie_links = {
    'Fresh': "https://lottie.host/f9acdb34-180b-41a6-8670-f579ae290779/V0fhWcwSKt.json",
    'Moderately Fresh': "https://lottie.host/158eabfc-4d8d-4964-a2d6-f61a4d7d86f7/XGVqULGrWu.json",
    'Spoiled': "https://lottie.host/49b79f94-2c7e-4685-a96c-5c999f351f2b/xkRaWhpAkz.json"
}
custom_messages = {
    'Fresh': "**✅ Fresh Fish Detected**\n\n- Less than 1 day old\n- Bright eyes, red gills\n- Perfect for cooking or raw dishes.",
    'Moderately Fresh': "**⚠️ Moderately Fresh**\n\n- 2–3 days old\n- Slight odor, dull eyes\n- Cook thoroughly before use.",
    'Spoiled': "**🚫 Spoiled Fish**\n\n- 4+ days old\n- Signs of decay, possible formalin\n- Unsafe to eat!"
}

# --- PAGE SELECTION ---
page = st.sidebar.selectbox("📂 Select Page", ["🏠 Welcome", "🔬 Prediction", "📞 Contact Us"])
st.sidebar.markdown("---")
st.sidebar.image("https://img.icons8.com/color/96/fish-food.png", width=100)
st.sidebar.info("Built  by **Jaydish Kennedy J**")

# --- WELCOME PAGE (Updated) ---
if page == "🏠 Welcome":
    st.title("🐟 Welcome to FishSnap App")
    st.subheader("AI-Powered Fish Freshness Detection for Safer Eating")

    st.markdown("""
    ### 🧪 Why This App Matters

    In many markets, **formalin** – a harmful chemical – is illegally used to preserve fish.  
    This can lead to:

    - ⚠️ **Food poisoning**
    - ❌ **Cancer risk**
    - 🧫 **Internal organ damage**

    ### 💡 How FishSnap Helps

    FishSnap uses a trained AI model to **analyze fish freshness from images**, identifying signs like:

    - 👁️ Cloudy or clear eyes  
    - 🟥 Gills coloration  
     

    ### ✅ What You Can Do

    1. Take a clear photo of the fish you intend to buy or consume  
    2. Upload it in the **Prediction** page  
    3. Let AI detect if it's **Fresh**, **Moderately Fresh**, or **Spoiled**

    ### 👥 Who Can Use This?

    - Home users  
    - Market inspectors  
    - Restaurant owners  
    - Food safety advocates

    ---
    Protect your family from hidden food hazards.  
    FishSnap is your quick **first line of defense against chemical contamination**.
    """)


# --- PREDICTION PAGE ---
elif page == "🔬 Prediction":
    st.title("🔍 Fish Freshness Prediction")
    uploaded_file = st.file_uploader("📤 Upload a fish image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="📸 Uploaded Image", use_column_width=True)

        # --- PROCESS IMAGE ---
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        input_data = np.expand_dims(img_array, axis=0)

        input_index = interpreter.get_input_details()[0]['index']
        output_index = interpreter.get_output_details()[0]['index']
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_index)

        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        predicted_class = class_names[predicted_index]

        # --- SHOW RESULTS ---
        st.success(f"🎯 Prediction: **{predicted_class}**")
        st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")
        st.info(custom_messages[predicted_class])

        # --- LOTTIE DISPLAY ---
        lottie_json = load_lottieurl(lottie_links[predicted_class])
        if lottie_json:
            components.html(f"""
                <lottie-player src="{lottie_links[predicted_class]}" background="transparent"
                               speed="1" style="width: 300px; height: 300px;" loop autoplay>
                </lottie-player>
                """, height=300)

        # --- SHOW CLASS PROBABILITIES ---
        st.subheader("📊 Prediction Breakdown")
        for i, class_name in enumerate(class_names):
            st.text(f"{class_name}: {prediction[0][i]*100:.2f}%")
            st.progress(float(np.clip(prediction[0][i], 0.0, 1.0)))

        # --- DOWNLOAD REPORT ---
        report_text = f"""Fish Freshness Prediction Report

Prediction: {predicted_class}
Confidence: {confidence * 100:.2f}%

{custom_messages[predicted_class]}
"""
        st.download_button("📥 Download Report", report_text, file_name="FishFreshnessReport.txt")

# --- CONTACT PAGE ---
elif page == "📞 Contact Us":
    st.title("📞 Contact Us")

    st.markdown("""
💬 **Have questions or feedback? We'd love to hear from you!**

---

### 👨‍💻 Developed By  
**Name:** Jaydish Kennedy J  
**Email:** [jaydishkennedy@example.com](mailto:jaydishkennedy@example.com)  
**Institution:** Department of Botany, St. Joseph’s College (Autonomous), Tiruchirappalli – 620002, Tamil Nadu, India  

---

### 👨‍🏫 Guide Information  
**Name:** Dr.G.Chelladurai. 
**Designation:** Assistant Professor of zoology  
**Department:** Botany  
**Institution:** St. Joseph’s College (Autonomous), Tiruchirappalli – 620002, Tamil Nadu, India  

---
""")

    with st.form(key='contact_form'):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message")
        submit_button = st.form_submit_button(label='📨 Send Message')

        if submit_button:
            st.success(f"✅ Thank you {name}, your message has been received.")
