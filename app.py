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

blood_pressure = st.number_input(
    "Tekanan Darah Sistolik (mmHg)",
    min_value=60,
    max_value=200,
    help="Tekanan darah normal dewasa < 120 mmHg"
)

chol = st.number_input(
    "Kolesterol Total (mg/dL)",
    min_value=100,
    max_value=400,
    help="Kolesterol total normal < 200 mg/dL"
)

st.divider()

# ======================
# Prediksi
# ======================
if st.button("🔍 Prediksi Risiko", use_container_width=True):

    # Prediksi Model ML
    data = np.array([[blood_pressure, chol]])
    pred = model.predict(data)

    st.subheader("📊 Hasil Prediksi")

    # ======================
    # Evaluasi Medis
    # ======================
    tekanan_normal = blood_pressure < 120
    kolesterol_normal = chol < 200

    if pred[0] == 1:
        st.error("⚠️ **BERISIKO Penyakit Jantung**")

        st.markdown("### 🔬 Analisis Medis:")
        if not tekanan_normal:
            st.markdown("- Tekanan darah berada di atas batas normal (<120 mmHg).")
        if not kolesterol_normal:
            st.markdown("- Kadar kolesterol total melebihi batas normal (<200 mg/dL).")

        st.markdown(
            "💡 **Saran:**\n"
            "- Lakukan pemeriksaan medis lanjutan\n"
            "- Jaga pola makan dan aktivitas fisik\n"
            "- Konsultasi dengan tenaga kesehatan"
        )

    else:
        st.success("✅ **TIDAK BERISIKO Penyakit Jantung**")

        st.markdown("### 🔬 Analisis Medis:")
        if tekanan_normal:
            st.markdown("- Tekanan darah berada dalam rentang normal.")
        else:
            st.markdown("- Tekanan darah perlu dikontrol meskipun hasil prediksi rendah.")

        if kolesterol_normal:
            st.markdown("- Kadar kolesterol total masih dalam batas normal.")
        else:
            st.markdown("- Kolesterol perlu diperhatikan meskipun risiko rendah.")

        st.markdown(
            "💡 **Saran:**\n"
            "- Pertahankan gaya hidup sehat\n"
            "- Lakukan pemeriksaan rutin\n"
            "- Jaga pola makan seimbang"
        )

# ======================
# Catatan Medis
# ======================
st.divider()
st.caption(
    "📌 Catatan: Tekanan darah normal dewasa <120 mmHg dan kolesterol total normal <200 mg/dL. "
    "Aplikasi ini bersifat pendukung keputusan dan tidak menggantikan diagnosis medis."
)
