import streamlit as st
import pickle
import numpy as np

# Judul aplikasi
st.title("🌸 Prediksi Kategori Bunga Iris dengan Neural Network")

# --- Load model ---
model_path = "iris_data.pkl"  # ganti dengan nama file model kamu
with open(model_path, "rb") as file:
    model = pickle.load(file)

st.sidebar.header("Masukkan Data Fitur")

# --- Input dari pengguna ---
sepal_length = st.sidebar.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.sidebar.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.sidebar.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.sidebar.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Tombol prediksi
if st.button("Prediksi Kategori"):
    # Ubah ke array numpy
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Prediksi
    prediction = model.predict(input_data)
    kelas = prediction[0]

    # Mapping hasil
    label = {
        0: "Iris-setosa 🌼",
        1: "Iris-versicolor 🌺",
        2: "Iris-virginica 🌸"
    }

    st.success(f"Hasil Prediksi: **{label.get(kelas, kelas)}**")

# Tambahkan footer
st.markdown("---")
st.caption("Dibuat oleh [Nama Kamu] menggunakan Streamlit & Orange Neural Network.")
