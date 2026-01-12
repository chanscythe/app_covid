import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Konfigurasi Halaman
st.set_page_config(page_title="Diagnosa COVID-19 Decision Tree", page_icon="🦠")

st.title("🦠 Sistem Pakar Diagnosa COVID-19")
st.markdown("""
Aplikasi ini menggunakan algoritma **Decision Tree** untuk mendiagnosa potensi COVID-19.
Konsep didasarkan pada pengubahan data gejala menjadi pohon keputusan dan aturan (rule).
""")

# --- 1. PREPARASI MODEL (Training Data) ---
@st.cache_resource
def build_model():
    # Data 20 Sampel sesuai tabel di Soal No 1
    data = {
        'Demam':       [1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1],
        'Batuk':       [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
        'Sesak':       [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        'Anosmia':     [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
        'Kontak':      [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
        'Diagnosa':    ['Positif', 'Positif', 'Negatif', 'Negatif', 'Positif', 'Positif', 
                        'Positif', 'Negatif', 'Positif', 'Positif', 'Positif', 'Positif',
                        'Negatif', 'Negatif', 'Positif', 'Positif', 'Positif', 'Negatif', 'Negatif', 'Positif']
    }
    df = pd.DataFrame(data)
    X = df[['Demam', 'Batuk', 'Sesak', 'Anosmia', 'Kontak']]
    y = df['Diagnosa']
    
    # Menggunakan Entropy sesuai materi kuliah
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X, y)
    return model

# Load Model
clf = build_model()

# --- 2. INPUT USER (SIDEBAR) ---
st.sidebar.header("Masukkan Gejala Pasien")
st.sidebar.info("Pilih kondisi yang dialami pasien.")

def get_user_input():
    demam = st.sidebar.radio("Apakah Demam?", ("Tidak", "Ya"))
    batuk = st.sidebar.radio("Apakah Batuk Kering?", ("Tidak", "Ya"))
    sesak = st.sidebar.radio("Apakah Sesak Nafas?", ("Tidak", "Ya"))
    anosmia = st.sidebar.radio("Apakah Kehilangan Penciuman (Anosmia)?", ("Tidak", "Ya"))
    kontak = st.sidebar.radio("Apakah Ada Kontak Erat?", ("Tidak", "Ya"))
    
    # Konversi ke numerik (0/1) agar bisa dibaca model
    input_data = {
        'Demam': 1 if demam == "Ya" else 0,
        'Batuk': 1 if batuk == "Ya" else 0,
        'Sesak': 1 if sesak == "Ya" else 0,
        'Anosmia': 1 if anosmia == "Ya" else 0,
        'Kontak': 1 if kontak == "Ya" else 0
    }
    return pd.DataFrame(input_data, index=[0])

user_input = get_user_input()

# --- 3. TAMPILAN UTAMA & PREDIKSI ---
st.subheader("Parameter Gejala Pasien:")
st.write(user_input)

if st.button("🔍 Analisa Diagnosa"):
    prediction = clf.predict(user_input)
    
    st.markdown("---")
    if prediction[0] == "Positif":
        st.error(f"### Hasil Diagnosa: {prediction[0]} COVID-19")
        st.write("Berdasarkan aturan Decision Tree, gejala ini mengarah pada indikasi Positif.")
        st.warning("Saran: Segera lakukan tes medis lanjutan (PCR) dan isolasi mandiri.")
    else:
        st.success(f"### Hasil Diagnosa: {prediction[0]} COVID-19")
        st.write("Gejala tidak menunjukkan indikasi kuat COVID-19 berdasarkan model.")
        st.info("Saran: Tetap jaga protokol kesehatan.")

# Footer
st.markdown("---")
st.caption("Referensi Metode: Dokumen SD-P9-20251117-Tree.pdf (Decision Tree & Rules)")