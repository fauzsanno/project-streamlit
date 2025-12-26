import streamlit as st
import joblib
import numpy as np

# ======================
# Page Config
# ======================
st.set_page_config(
    page_title="Prediksi Risiko Penyakit Jantung",
    page_icon="❤️",
    layout="centered"
)

# ======================
# Styling
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
# Load Model & Scaler
# ======================
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ======================
# Header
# ======================
st.title("❤️ Prediksi Risiko Penyakit Jantung")
st.write(
    "Aplikasi ini memprediksi risiko penyakit jantung "
    "berdasarkan data kesehatan menggunakan Machine Learning."
)

st.divider()

# ======================
# Input User
# ======================
st.subheader("🩺 Data Kesehatan Pasien")

age = st.number_input("Usia (hari)", 1000, 30000, 18000)
gender = st.selectbox("Jenis Kelamin", [1, 2], format_func=lambda x: "Wanita" if x == 1 else "Pria")
height = st.number_input("Tinggi Badan (cm)", 100, 250, 170)
weight = st.number_input("Berat Badan (kg)", 30, 200, 70)
ap_hi = st.number_input("Tekanan Darah Sistolik (mmHg)", 70, 200, 120)
ap_lo = st.number_input("Tekanan Darah Diastolik (mmHg)", 40, 130, 80)
cholesterol = st.selectbox("Kolesterol (1=Normal, 2–3=Tinggi)", [1, 2, 3])
gluc = st.selectbox("Glukosa (1=Normal, 2–3=Tinggi)", [1, 2, 3])
smoke = st.selectbox("Merokok", [0, 1])
alco = st.selectbox("Konsumsi Alkohol", [0, 1])
active = st.selectbox("Aktivitas Fisik", [0, 1])

st.divider()

# ======================
# Prediction
# ======================
if st.button("🔍 Prediksi Risiko", use_container_width=True):

    # URUTAN HARUS SAMA PERSIS DENGAN DATASET
    input_array = np.array([[
        age,
        gender,
        height,
        weight,
        ap_hi,
        ap_lo,
        cholesterol,
        gluc,
        smoke,
        alco,
        active
    ]])

    # Scaling (AMAN, TANPA FEATURE NAME ERROR)
    input_scaled = scaler.transform(input_array)

    # Prediction
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Hasil Prediksi")

    if pred == 1:
        st.error(f"⚠️ **BERISIKO Penyakit Jantung**\n\nProbabilitas: **{prob:.2%}**")
    else:
        st.success(f"✅ **TIDAK BERISIKO Penyakit Jantung**\n\nProbabilitas: **{prob:.2%}**")

# ======================
# Notes
# ======================
st.divider()
st.caption(
    "📌 Catatan:\n"
    "- Aplikasi ini adalah alat bantu keputusan\n"
    "- Tidak menggantikan diagnosis dokter"
)
