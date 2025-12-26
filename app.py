import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ======================
# Styling Background
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
# Konfigurasi Halaman
# ======================
st.set_page_config(
    page_title="Prediksi Risiko Penyakit Jantung",
    page_icon="❤️",
    layout="centered"
)

# ======================
# Load Model & Preprocessing
# ======================
@st.cache_resource
def load_all():
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.pkl")
    selector = joblib.load("selector.pkl")
    selected_features = joblib.load("selected_features.pkl")
    return model, scaler, selector, selected_features

model, scaler, selector, selected_features = load_all()

# ======================
# Header
# ======================
st.title("❤️ Prediksi Risiko Penyakit Jantung")
st.write(
    "Aplikasi ini memprediksi risiko penyakit jantung menggunakan "
    "Machine Learning (XGBoost + LightGBM) dan evaluasi medis standar."
)

st.divider()

# ======================
# Input User
# ======================
st.subheader("🩺 Data Kesehatan Pasien")

age = st.number_input("Usia (hari)", 1000, 30000, 18000)
gender = st.selectbox(
    "Jenis Kelamin",
    options=[1, 2],
    format_func=lambda x: "Wanita" if x == 1 else "Pria"
)
height = st.number_input("Tinggi Badan (cm)", 100, 250, 170)
weight = st.number_input("Berat Badan (kg)", 30, 200, 70)

sistolik = st.number_input(
    "Tekanan Darah Sistolik (mmHg)",
    70, 200, 120,
    help="Normal: <120 mmHg"
)

diastolik = st.number_input(
    "Tekanan Darah Diastolik (mmHg)",
    40, 130, 80,
    help="Normal: ≥60 mmHg"
)

cholesterol = st.selectbox("Kolesterol (1=Normal, 2–3=Tinggi)", [1, 2, 3])
gluc = st.selectbox("Glukosa (1=Normal, 2–3=Tinggi)", [1, 2, 3])
smoke = st.selectbox("Merokok", [0, 1])
alco = st.selectbox("Konsumsi Alkohol", [0, 1])
active = st.selectbox("Aktivitas Fisik", [0, 1])

st.divider()

# ======================
# Prediksi
# ======================
if st.button("🔍 Prediksi Risiko", use_container_width=True):

    # ======================
    # DataFrame Input
    # ======================
    input_df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": sistolik,
        "ap_lo": diastolik,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active
    }])

    # ======================
    # Feature Engineering
    # ======================
    input_df["BMI"] = input_df["weight"] / ((input_df["height"] / 100) ** 2)
    input_df["pressure_diff"] = input_df["ap_hi"] - input_df["ap_lo"]

    # ======================
    # Urutan fitur WAJIB SAMA
    # ======================
    feature_order = [
        "age", "gender", "height", "weight",
        "ap_hi", "ap_lo",
        "cholesterol", "gluc",
        "smoke", "alco", "active",
        "BMI", "pressure_diff"
    ]
    input_df = input_df[feature_order]

    # ======================
    # Preprocessing
    # ======================
    input_scaled = scaler.transform(input_df)
    input_selected = selector.transform(input_scaled)

    # 🔥 FIX FINAL (WAJIB UNTUK XGBOOST)
    input_selected = np.asarray(input_selected)

    # ======================
    # Prediksi Model
    # ======================
    pred = model.predict(input_selected)[0]
    prob = model.predict_proba(input_selected)[0][1]

    st.subheader("📊 Hasil Prediksi")

    # ======================
    # Evaluasi Medis (Rule-Based)
    # ======================
    tekanan_normal = (sistolik < 120) and (diastolik >= 60)
    kolesterol_normal = cholesterol == 1

    if pred == 1:
        st.error(f"⚠️ **BERISIKO Penyakit Jantung**\n\nProbabilitas: **{prob:.2%}**")

        st.markdown("### 🔬 Analisis Medis:")
        if sistolik >= 120:
            st.markdown("- Tekanan darah sistolik di atas batas normal.")
        if diastolik < 60:
            st.markdown("- Tekanan darah diastolik terlalu rendah (hipotensi).")
        if not kolesterol_normal:
            st.markdown("- Kadar kolesterol melebihi batas normal.")

        st.markdown(
            "💡 **Saran Medis:**\n"
            "- Lakukan pemeriksaan medis lanjutan\n"
            "- Kendalikan tekanan darah dan kolesterol\n"
            "- Konsultasi dengan dokter"
        )
    else:
        st.success(f"✅ **TIDAK BERISIKO Penyakit Jantung**\n\nProbabilitas: **{prob:.2%}**")

        st.markdown("### 🔬 Analisis Medis:")
        if tekanan_normal:
            st.markdown("- Tekanan darah berada dalam rentang normal.")
        else:
            st.markdown("- Tekanan darah perlu dipantau.")

        if kolesterol_normal:
            st.markdown("- Kolesterol berada dalam batas normal.")
        else:
            st.markdown("- Kolesterol perlu dikontrol.")

        st.markdown(
            "💡 **Saran Medis:**\n"
            "- Pertahankan pola hidup sehat\n"
            "- Rutin pemeriksaan kesehatan\n"
            "- Jaga pola makan dan aktivitas fisik"
        )

# ======================
# Catatan Medis
# ======================
st.divider()
st.caption(
    "📌 Catatan Medis:\n"
    "- Tekanan darah normal dewasa: sistolik <120 mmHg dan diastolik ≥60 mmHg\n"
    "- Kolesterol normal: level 1\n"
    "Aplikasi ini bersifat pendukung keputusan dan tidak menggantikan diagnosis medis."
)
