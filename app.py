import streamlit as st
import joblib
import numpy as np

st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e88e5;
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
# Load Model
# ======================
model = joblib.load("model.pkl")

# ======================
# Header
# ======================
st.title("❤️ Prediksi Risiko Penyakit Jantung")
st.write(
    "Aplikasi ini memprediksi risiko penyakit jantung menggunakan model "
    "Machine Learning serta mempertimbangkan standar medis tekanan darah dan kolesterol."
)

st.divider()

# ======================
# Input User
# ======================
st.subheader("🩺 Data Kesehatan")

sistolik = st.number_input(
    "Tekanan Darah Sistolik (mmHg)",
    min_value=70,
    max_value=200,
    value=120,
    help="Normal: < 120 mmHg"
)

diastolik = st.number_input(
    "Tekanan Darah Diastolik (mmHg)",
    min_value=40,
    max_value=130,
    value=80,
    help="Normal: ≥ 60 mmHg"
)

chol = st.number_input(
    "Kolesterol Total (mg/dL)",
    min_value=100,
    max_value=400,
    value=180,
    help="Normal: < 200 mg/dL"
)

st.divider()

# ======================
# Prediksi
# ======================
if st.button("🔍 Prediksi Risiko", use_container_width=True):

    # ======================
    # Prediksi Model ML
    # ======================
    data = np.array([[sistolik, diastolik, chol]])
    pred = model.predict(data)

    st.subheader("📊 Hasil Prediksi")

    # ======================
    # Evaluasi Medis (Rule-Based)
    # ======================
    tekanan_normal = (sistolik < 120) and (diastolik >= 60)
    kolesterol_normal = chol < 200

    if pred[0] == 1:
        st.error("⚠️ **BERISIKO Penyakit Jantung**")

        st.markdown("### 🔬 Analisis Medis:")
        if sistolik >= 120:
            st.markdown("- Tekanan darah sistolik berada di atas batas normal (<120 mmHg).")
        if diastolik < 60:
            st.markdown("- Tekanan darah diastolik terlalu rendah (<60 mmHg / hipotensi).")
        if not kolesterol_normal:
            st.markdown("- Kadar kolesterol total melebihi batas normal (<200 mg/dL).")

        st.markdown(
            "💡 **Saran Medis:**\n"
            "- Lakukan pemeriksaan medis lanjutan\n"
            "- Kendalikan tekanan darah dan kolesterol\n"
            "- Konsultasi dengan dokter atau tenaga kesehatan"
        )

    else:
        st.success("✅ **TIDAK BERISIKO Penyakit Jantung**")

        st.markdown("### 🔬 Analisis Medis:")
        if tekanan_normal:
            st.markdown("- Tekanan darah berada dalam rentang normal.")
        else:
            st.markdown("- Tekanan darah perlu dipantau meskipun risiko rendah.")

        if kolesterol_normal:
            st.markdown("- Kadar kolesterol total berada dalam batas normal.")
        else:
            st.markdown("- Kolesterol perlu dikontrol untuk mencegah risiko di masa depan.")

        st.markdown(
            "💡 **Saran Medis:**\n"
            "- Pertahankan pola hidup sehat\n"
            "- Rutin melakukan pemeriksaan kesehatan\n"
            "- Jaga pola makan dan aktivitas fisik"
        )

# ======================
# Catatan Medis
# ======================
st.divider()
st.caption(
    "📌 Catatan Medis:\n"
    "- Tekanan darah normal dewasa: sistolik <120 mmHg dan diastolik ≥60 mmHg\n"
    "- Kolesterol total normal: <200 mg/dL\n"
    "Aplikasi ini bersifat pendukung keputusan dan tidak menggantikan diagnosis medis."
)

