import streamlit as st
import joblib
import numpy as np

# ======================
# Konfigurasi Halaman
# ======================
st.set_page_config(
    page_title="Prediksi Risiko Penyakit Jantung",
    page_icon="❤️",
    layout="centered"
)

# ======================
# Load Model
# ======================
model = joblib.load("model.pkl")

# ======================
# Header
# ======================
st.markdown(
    "<h1 style='text-align: center; color: #d63031;'>Prediksi Risiko Penyakit Jantung</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Masukkan data kesehatan untuk mengetahui risiko penyakit jantung</p>",
    unsafe_allow_html=True
)

st.divider()

# ======================
# Input Section
# ======================
st.subheader("🩺 Data Pasien")

col1, col2 = st.columns(2)

with col1:
    blood_pressure = st.number_input(
        "Tekanan Darah (mmHg)",
        min_value=60,
        max_value=200,
        help="Masukkan tekanan darah sistolik"
    )

with col2:
    chol = st.number_input(
        "Kolesterol (mg/dL)",
        min_value=100,
        max_value=400,
        help="Masukkan kadar kolesterol total"
    )

st.divider()

# ======================
# Prediction Button
# ======================
if st.button("🔍 Prediksi Risiko", use_container_width=True):

    data = np.array([[blood_pressure, chol]])
    pred = model.predict(data)

    st.subheader("📊 Hasil Prediksi")

    if pred[0] == 1:
        st.error("⚠️ **Berisiko Penyakit Jantung**")
        st.markdown(
            "- Disarankan melakukan pemeriksaan medis lebih lanjut\n"
            "- Menjaga pola hidup sehat\n"
            "- Konsultasi dengan tenaga kesehatan"
        )
    else:
        st.success("✅ **Tidak Berisiko Penyakit Jantung**")
        st.markdown(
            "- Pertahankan pola hidup sehat\n"
            "- Rutin berolahraga\n"
            "- Lakukan pemeriksaan berkala"
        )

# ======================
# Footer
# ======================
st.markdown(
    "<hr><p style='text-align:center; font-size:12px;'>"
    "Aplikasi ini menggunakan model Machine Learning untuk tujuan edukasi"
    "</p>",
    unsafe_allow_html=True
)
