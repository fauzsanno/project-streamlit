import streamlit as st
import joblib
import pandas as pd

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
# Konfigurasi Halaman
# ======================
st.set_page_config(
    page_title="Prediksi Risiko Penyakit Jantung",
    page_icon="❤️",
    layout="centered"
)

# ======================
# Load PIPELINE (WAJIB)
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
    "Aplikasi ini memprediksi risiko penyakit jantung menggunakan model "
    "Machine Learning berbasis data medis."
)

st.divider()

# ======================
# Input User (SESUAI DATASET)
# ======================
st.subheader("🩺 Data Kesehatan")

age = st.number_input(
    "Usia (dalam hari)",
    min_value=0,
    value=18000,
    help="Contoh: 50 tahun ≈ 18250 hari"
)

gender = st.selectbox(
    "Jenis Kelamin",
    options=[1, 2],
    format_func=lambda x: "Perempuan" if x == 1 else "Laki-laki"
)

height = st.number_input(
    "Tinggi Badan (cm)",
    min_value=100,
    max_value=220,
    value=165
)

weight = st.number_input(
    "Berat Badan (kg)",
    min_value=30,
    max_value=200,
    value=65
)

ap_hi = st.number_input(
    "Tekanan Darah Sistolik (mmHg)",
    min_value=70,
    max_value=250,
    value=120
)

ap_lo = st.number_input(
    "Tekanan Darah Diastolik (mmHg)",
    min_value=40,
    max_value=200,
    value=80
)

cholesterol = st.selectbox(
    "Kolesterol",
    options=[1, 2, 3],
    format_func=lambda x: ["Normal", "Di atas normal", "Sangat tinggi"][x - 1]
)

gluc = st.selectbox(
    "Glukosa",
    options=[1, 2, 3],
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

  # ======================
# Prediksi
# ======================
if st.button("🔍 Prediksi Risiko", use_container_width=True):

    # URUTAN WAJIB SAMA DENGAN CSV TRAINING
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

    # PREDIKSI
    pred = model.predict(input_array)[0]
    prob = model.predict_proba(input_array)[0][1]


    st.subheader("📊 Hasil Prediksi")

    if pred == 1:
        st.error(f"⚠️ **BERISIKO Penyakit Jantung** ({prob:.2%})")

        st.markdown("### 🔬 Analisis:")
        if ap_hi >= 120:
            st.markdown("- Tekanan darah sistolik di atas normal.")
        if ap_lo < 60:
            st.markdown("- Tekanan darah diastolik rendah (hipotensi).")
        if cholesterol > 1:
            st.markdown("- Kadar kolesterol tidak normal.")

        st.markdown(
            "💡 **Saran:**\n"
            "- Lakukan pemeriksaan medis lanjutan\n"
            "- Kontrol tekanan darah dan kolesterol\n"
            "- Konsultasi dengan dokter"
        )

    else:
        st.success(f"✅ **TIDAK BERISIKO Penyakit Jantung** ({1 - prob:.2%})")

        st.markdown("### 🔬 Analisis:")
        st.markdown("- Parameter kesehatan relatif stabil.")

        st.markdown(
            "💡 **Saran:**\n"
            "- Pertahankan pola hidup sehat\n"
            "- Rutin berolahraga dan cek kesehatan"
        )

# ======================
# Catatan Medis
# ======================
st.divider()
st.caption(
    "📌 **Catatan:**\n"
    "- Aplikasi ini adalah sistem pendukung keputusan\n"
    "- Tidak menggantikan diagnosis dokter\n"
    "- Konsultasikan hasil dengan tenaga medis profesional"
)






