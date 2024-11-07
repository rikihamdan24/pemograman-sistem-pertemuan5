import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('/workspaces/pemograman-sistem-pertemuan5/Android_Malware_Benign.csv')

# Pilih subset fitur yang diinginkan
selected_features = [
    'ACCESS_ALL_DOWNLOADS', 'ACCESS_CACHE_FILESYSTEM', 'ACCESS_COARSE_LOCATION',
    'ACCESS_FINE_LOCATION', 'ACCESS_NETWORK_STATE', 'android.permission.READ_SMS'
]

# Pastikan fitur yang dipilih ada di dataset
X = df[selected_features]  # Gunakan kolom yang dipilih untuk fitur
y = df['Label']  # Kolom target

# Konversi target ke label numerik jika diperlukan
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Membagi dataset menjadi data train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Melatih model
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Mengecek akurasi model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi model dengan fitur yang dipilih: {accuracy}")

# Menyimpan model dan label encoder ke file
with open("classifier_selected_features.pkl", "wb") as model_file:
    pickle.dump(classifier, model_file)

with open("label_encoder.pkl", "wb") as encoder_file:
    pickle.dump(label_encoder, encoder_file)

import streamlit as st
import pickle

# Memuat model dan label encoder
with open("classifier_selected_features.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Fungsi prediksi menggunakan model
def prediction(features):
    pred = classifier.predict([features])
    return pred

# Fungsi untuk mengonversi hasil prediksi ke label yang sesuai
def map_prediction_to_class(prediction):
    # Mengembalikan label asli, misalnya 'Malicious' atau 'Benign'
    return label_encoder.inverse_transform([prediction])[0]

# Streamlit
def main():
    # Judul halaman
    st.title("Malware Classification Prediction")

    # Desain halaman dengan HTML dan tambahan nama
    html_temp = """
    <div style="background-color:yellow;padding:13px">
    <h1 style="color:black;text-align:center;">Malware Classifier ML App</h1>
    <h2 style="color:blue;text-align:center;">Riki Hamdan Sucipto 22220038</h4>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Input dari user (fitur sesuai dataset)
    access_all_downloads = st.number_input("ACCESS_ALL_DOWNLOADS", value=0)
    access_cache_filesystem = st.number_input("ACCESS_CACHE_FILESYSTEM", value=0)
    access_coarse_location = st.number_input("ACCESS_COARSE_LOCATION", value=0)
    access_fine_location = st.number_input("ACCESS_FINE_LOCATION", value=0)
    access_network_state = st.number_input("ACCESS_NETWORK_STATE", value=0)
    read_sms = st.number_input("android.permission.READ_SMS", value=0)

    # Menggabungkan input menjadi satu list untuk prediksi
    features = [
        access_all_downloads, access_cache_filesystem, access_coarse_location,
        access_fine_location, access_network_state, read_sms
    ]

    result = ""

    # Tombol prediksi
    if st.button("Predict"):
        # Melakukan prediksi
        prediction_result = prediction(features)

        # Konversi hasil prediksi ke label asli (Malicious atau Benign)
        result = map_prediction_to_class(prediction_result[0])

    st.success(f'The system is classified as: {result}')

if __name__ == '__main__':
    main()
