import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import pytesseract
import os

# -------- CONFIG --------
st.set_page_config(
    page_title="FactCheck AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------- OCR PATH --------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------- LOAD MODEL --------
model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

DATA_FILE = "combined_news.csv"

# Create CSV if not exists
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=["text", "prediction"]).to_csv(DATA_FILE, index=False)

# -------- CSS --------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -------- SIDEBAR (HAMBURGER MENU) --------
st.sidebar.image("logo.jpg", use_container_width=True)
st.sidebar.markdown("## 🧠 FactCheck AI")

menu = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Home", "🔍 Verify News", "📊 Dashboard"]
)

# ================= HOME =================
if menu == "🏠 Home":

    st.markdown("""
    <h1 style='text-align:center;'>🧠 FactCheck AI</h1>
    <p style='text-align:center; font-size:18px;'>
    Detect Fake News instantly using Machine Learning 🚀
    </p>
    """, unsafe_allow_html=True)

    st.image("bg.jpg", use_container_width=True)

    st.markdown("---")

    st.markdown("## 🔍 How It Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📝 Enter News")
        st.write("Paste news or upload image")

    with col2:
        st.markdown("### 🤖 AI Analysis")
        st.write("Model processes the content")

    with col3:
        st.markdown("### ✅ Get Result")
        st.write("Get Real/Fake prediction")

    st.markdown("---")

    st.markdown("## ⭐ Why Use This App?")
    st.write("""
    ✔ Fast fake news detection  
    ✔ Works on real-time news  
    ✔ Supports text & image  
    ✔ Simple & user-friendly  
    """)

# ================= VERIFY =================
elif menu == "🔍 Verify News":

    st.markdown("## 🧪 Verify Your News")
    st.caption("Enter text OR upload image")

    # Sample button
    if st.button("Try Sample News"):
        text_input = "Breaking: Government launches new student scheme"
    else:
        text_input = st.text_area("Enter News", height=120)

    uploaded_file = st.file_uploader("Upload Image (optional)", type=["png","jpg","jpeg"])

    ocr_text = ""

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        ocr_text = pytesseract.image_to_string(image)

    if st.button("Predict"):

        if not text_input and not ocr_text:
            st.warning("⚠️ Enter text or upload image")
        else:
            news_text = text_input if text_input else ocr_text

            vect = vectorizer.transform([news_text])
            prediction = model.predict(vect)[0]
            proba = model.predict_proba(vect)[0]
            confidence = max(proba) * 100

            st.markdown("---")

            if prediction == 1:
                st.success("✅ Real News 😊")
                result = "REAL"
            else:
                st.error("❌ Fake News 😡")
                result = "FAKE"

            st.progress(int(confidence))
            st.write(f"Confidence: {confidence:.2f}%")

            # Save data
            df = pd.read_csv(DATA_FILE)
            new_row = pd.DataFrame([{"text": news_text, "prediction": result}])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)

# ================= DASHBOARD =================
elif menu == "📊 Dashboard":

    st.markdown("## 📊 Dashboard")

    df = pd.read_csv(DATA_FILE)

    if len(df) == 0:
        st.warning("No data yet")
    else:
        total = len(df)
        real = len(df[df["prediction"] == "REAL"])
        fake = len(df[df["prediction"] == "FAKE"])

        col1, col2, col3 = st.columns(3)

        col1.metric("Total", total)
        col2.metric("Real", real)
        col3.metric("Fake", fake)

        st.markdown("---")

        st.subheader("Distribution")
        st.bar_chart(df["prediction"].value_counts())

        st.subheader("Recent Predictions")
        st.dataframe(df.tail(5))

    # -------- FEEDBACK --------
    st.markdown("---")
    st.markdown("## ✉️ Feedback")

    name = st.text_input("Name")
    message = st.text_area("Message")

    if st.button("Submit Feedback"):
        if name and message:
            fb_file = "feedback.csv"

            if os.path.exists(fb_file):
                fb = pd.read_csv(fb_file)
            else:
                fb = pd.DataFrame(columns=["name", "message"])

            new_fb = pd.DataFrame([{"name": name, "message": message}])
            fb = pd.concat([fb, new_fb], ignore_index=True)
            fb.to_csv(fb_file, index=False)

            st.success("Thanks for feedback! ✅")
        else:
            st.warning("Fill all fields")