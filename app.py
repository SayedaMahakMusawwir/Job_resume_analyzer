import streamlit as st
import pickle
import pdfplumber
import docx
import re
import pandas as pd
import requests
import os
from io import StringIO

# ------------------ DOWNLOAD FROM GOOGLE DRIVE ------------------
def download_from_gdrive(file_id, dest_path):
    """Download a file from Google Drive if it doesn't exist locally."""
    if not os.path.exists(dest_path):
        st.info(f"⬇️ Downloading {dest_path} from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        r = requests.get(url)
        with open(dest_path, "wb") as f:
            f.write(r.content)
        st.success(f"{dest_path} downloaded successfully!")

# ⚠️  REPLACE THESE IDs with your actual Google Drive file IDs
MODEL_ID = "1AbCdEfGhIjKlMnOpQr"        # knn_model.pkl
VECTORIZER_ID = "2ZyXwVuTsRqPoNmLkJi"   # vectorizer.pkl
ENCODER_ID = "3FeDcBaQwErTyUiOpLk"      # label_encoder.pkl

@st.cache_resource
def load_all():
    download_from_gdrive(MODEL_ID, "knn_model.pkl")
    download_from_gdrive(VECTORIZER_ID, "vectorizer.pkl")
    download_from_gdrive(ENCODER_ID, "label_encoder.pkl")

    model = pickle.load(open("knn_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_all()

# ------------------ CLEANING ------------------
def clean_resume(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = " ".join(text.split())
    return text

# ------------------ EXTRACT TEXT ------------------
def extract_text(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    elif uploaded_file.name.lower().endswith(".docx"):
        document = docx.Document(uploaded_file)
        return " ".join([p.text for p in document.paragraphs])
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="Resume Category Predictor", layout="centered")
st.title("📄 Resume Category Prediction App")
st.write("Upload a resume (PDF/DOCX/TXT) — this app predicts its **job category** using your trained **KNN model**.")

uploaded = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

if uploaded is not None:
    st.info(f"File uploaded: {uploaded.name}")
    resume_text = extract_text(uploaded)

    if len(resume_text.strip()) == 0:
        st.warning("Couldn't extract text. Try another file or ensure it’s not an image-only PDF.")
    else:
        st.subheader("Preview of Extracted Text:")
        st.text(resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)

        if st.button("🔍 Predict Category"):
            with st.spinner("Analyzing..."):
                cleaned = clean_resume(resume_text)
                vector = vectorizer.transform([cleaned])
                pred_label_num = model.predict(vector)[0]
                pred_label_name = label_encoder.inverse_transform([pred_label_num])[0]

            st.success("✅ Prediction Complete!")
            st.markdown(f"""
            **Category Number:** `{pred_label_num}`  
            **Category Name:** 🎯 **{pred_label_name}**
            """)

            # ---------- CSV DOWNLOAD FEATURE ----------
            result_df = pd.DataFrame({
                "Resume File": [uploaded.name],
                "Category Number": [pred_label_num],
                "Category Name": [pred_label_name]
            })
            csv_buffer = StringIO()
            result_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="📥 Download Prediction as CSV",
                data=csv_data,
                file_name=f"{uploaded.name.split('.')[0]}_prediction.csv",
                mime="text/csv"
            )

# ------------------ SIDEBAR INFO ------------------
st.sidebar.title("ℹ️ About This App")
st.sidebar.write("""
**Resume Category Prediction App**

- Input: Resume (PDF/DOCX/TXT)  
- Output: Predicted Job Category Name and Encoded Number  
- Model: K-Nearest Neighbors (KNN)  
- Hosted model files on Google Drive  

Developed for educational purposes 💡
""")
