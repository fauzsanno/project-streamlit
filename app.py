import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ======================
# Konfigurasi Halaman
# ======================
st.set_page_config(
    page_title="Prediksi Risiko Penyakit Jantung",
    page_icon="❤️",
    layout="centered"
)

# ======================
# Style Background
# ======================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #D9EBFA;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================
# Load PIPELINE (pipeline lengkap)
# ======================
@st.cache_resource
def load_model():
    return joblib.load("pipeline.joblib")

model = load_model()

# ======================
# Header
# ======================
st.title("❤️ Prediksi Risiko Penyakit Jantung")
st.write(
    "Aplikasi ini memprediksi risiko penyakit jantung menggunakan "
    "model Machine Learning berbasis data medis."
)

st.divider()

# ======================
# Input User
# ======================
st.subheader("🩺 Data Kesehatan")

age = st.number_input("Usia (hari)", min_value=0, value=18000)
gender = st.selectbox("Jenis Kelamin", [1, 2],
                      format_func=lambda x: "Perempuan" if x == 1 else "Laki-laki")
height = st.number_input("Tinggi Badan (cm)", 100, 220, 165)
weight = st.number_input("Berat Badan (kg)", 30, 200, 65)
ap_hi = st.number_input("Tekanan Darah Sistolik", 70, 250, 120)
ap_lo = st.number_input("Tekanan Darah Diastolik", 40, 200, 80)

cholesterol = st.selectbox(
    "Kolesterol", [1, 2, 3],
    format_func=lambda x: ["Normal", "Di atas normal", "Sangat tinggi"][x - 1]
)

gluc = st.selectbox(
    "Glukosa", [1, 2, 3],
    format_func=lambda x: ["Normal", "Di atas normal", "Sangat tinggi"][x - 1]
)

smoke = st.selectbox("Merokok", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
alco = st.selectbox("Konsumsi Alkohol", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
active = st.selectbox("Aktivitas Fisik", [0, 1], format_func=lambda x: "Tidak Aktif" if x == 0 else "Aktif")

st.divider()

# ======================
# Prediksi
# ======================
if st.button("🔍 Prediksi Risiko", use_container_width=True):

    # DataFrame dengan FEATURE NAME RESMI
    input_df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active
    }])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Hasil Prediksi")

    if pred == 1:
        st.error(f"⚠️ **BERISIKO Penyakit Jantung** ({prob:.2%})")
    else:
        st.success(f"✅ **TIDAK BERISIKO Penyakit Jantung** ({1 - prob:.2%})")

# ======================
# Catatan Medis
# ======================
st.divider()
st.caption(
    "📌 Aplikasi ini adalah sistem pendukung keputusan dan "
    "tidak menggantikan diagnosis dokter."
)
