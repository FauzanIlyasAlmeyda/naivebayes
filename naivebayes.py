import pandas as pd
import numpy as np
from math import log

# Data Latih
data_latih = [
    ['>=3', 'Baik', 'Baik', '>=3', 'Ya'],
    ['>=3', 'Baik', 'Baik', '>=3', 'Ya'],
    ['>=3', 'Baik', 'Baik', '>=3', 'Ya'],
    ['>=3', 'Baik', 'Baik', '>=3', 'Ya'],
    ['>=3', 'Baik', 'Baik', '>=3', 'Ya'],
    ['>=3', 'Baik', 'Baik', '>=3', 'Ya'],
    ['>=3', 'Baik', 'Baik', '>=3', 'Ya'],
    ['>=3', 'Baik', 'Baik', '>=3', 'Ya'],
    ['>=3', 'Kurang Baik', 'Baik', '>=3', 'Tidak'],
    ['>=3', 'Kurang Baik', 'Baik', '>=3', 'Tidak'],
    ['>=3', 'Kurang Baik', 'Kurang Baik', '>=3', 'Tidak'],
    ['>=3', 'Kurang Baik', 'Baik', '>=3', 'Tidak'],
    ['>=3', 'Kurang Baik', 'Baik', '>=3', 'Tidak'],
    ['>=3', 'Kurang Baik', 'Kurang Baik', '>=3', 'Tidak'],
    ['>=3', 'Kurang Baik', 'Kurang Baik', '>=3', 'Tidak'],
    ['>=3', 'Baik', 'Baik', '>=3', 'Ya'],
    ['>=3', 'Baik', 'Baik', '>=3', 'Ya'],
    ['>=3', 'Baik', 'Baik', '>=3', 'Ya'],
    ['>=3', 'Baik', 'Baik', '>=3', 'Ya'],
    ['>=3', 'Baik', 'Baik', '>=3', 'Ya']
]

# Data Uji
data_uji = [
    ['BIG-00085', 'Juwita Putri', '>=3', 'Baik', 'Baik', '>=3'],
    ['BIG-00086', 'Kunto Adi', '>=3', 'Baik', 'Baik', '>=3'],
    ['BIG-00087', 'Lilis Setiyowati', '>=3', 'Baik', 'Baik', '>=3'],
    ['BIG-00088', 'Maman Suryadi', '>=3', 'Baik', 'Baik', '>=3'],
    ['BIG-00089', 'Nina Hartati', '>=3', 'Baik', 'Baik', '>=3'],
    ['BIG-00090', 'Oni Supriyadi', '>=3', 'Baik', 'Baik', '>=3'],
    ['BIG-00091', 'Puput Lestari', '>=3', 'Baik', 'Baik', '>=3'],
    ['BIG-00092', 'Qori Prasetyo', '>=3', 'Baik', 'Baik', '>=3'],
    ['BIG-00093', 'Reni Puspitasari', '>=3', 'Baik', 'Baik', '>=3'],
    ['BIG-00094', 'Seno Wibisono', '>=3', 'Baik', 'Baik', '>=3'],
    ['BIG-00095', 'Tia Nurhaliza', '>=3', 'Baik', 'Baik', '>=3'],
    ['BIG-00096', 'Untung Handoko', '>=3', 'Baik', 'Baik', '>=3'],
    ['BIG-00097', 'Vina Wijayanti', '>=3', 'Baik', 'Baik', '>=3'],
    ['BIG-00098', 'Wawan Hidayat', '>=3', 'Baik', 'Baik', '>=3'],
    ['BIG-00099', 'Yana Pratiwi', '>=3', 'Kurang Baik', 'Baik', '>=3'],
    ['BIG-00100', 'Zahid Maulana', '>=3', 'Kurang Baik', 'Baik', '>=3'],
    ['BIG-00101', 'Ahmad Fauzi', '>=3', 'Baik', 'Kurang Baik', '<3'],
    ['BIG-00102', 'Bunga Pertiwi', '<3', 'Baik', 'Baik', '>=3'],
    ['BIG-00103', 'Chandra Dewi', '<3', 'Baik', 'Baik', '>=3'],
    ['BIG-00104', 'Deni Maulana', '>=3', 'Kurang Baik', 'Baik', '>=3'],
    ['BIG-00105', 'Elsa Putri', '>=3', 'Kurang Baik', 'Kurang Baik', '>=3'],
    ['BIG-00106', 'Farid Rahman', '>=3', 'Kurang Baik', 'Kurang Baik', '<3'],
    ['BIG-00107', 'Gina Kurniawati', '>=3', 'Kurang Baik', 'Kurang Baik', '<3'],
    ['BIG-00108', 'Hadi Saputra', '>=3', 'Kurang Baik', 'Kurang Baik', '<3']
]

# Konversi data ke numerik
def konversi_ke_numerik(data):
    data_numerik = []
    for baris in data:
        baris_baru = []
        for i, nilai in enumerate(baris):
            if i < len(baris) - 1:  # Proses fitur
                if nilai == '>=3':
                    baris_baru.append(1)
                elif nilai == '<3':
                    baris_baru.append(0)
                elif nilai == 'Baik':
                    baris_baru.append(1)
                elif nilai == 'Kurang Baik':
                    baris_baru.append(0)
            else:  # Proses label
                if nilai == 'Ya':
                    baris_baru.append(1)
                elif nilai == 'Tidak':
                    baris_baru.append(0)
        data_numerik.append(baris_baru)
    return data_numerik

# Mengonversi data latih dan uji
data_latih_numerik = konversi_ke_numerik(data_latih)
data_uji_numerik = [konversi_ke_numerik([baris[2:]])[0] for baris in data_uji]

# Fungsi untuk menghitung probabilitas kelas
def hitung_probabilitas_kelas(data_latih_numerik):
    probabilitas_kelas = {}
    total_data = len(data_latih_numerik)
    for label in [0, 1]:  # 0 = Tidak, 1 = Ya
        probabilitas_kelas[label] = sum([1 for baris in data_latih_numerik if baris[-1] == label]) / total_data
    return probabilitas_kelas

# Fungsi untuk menghitung probabilitas fitur
def hitung_probabilitas_fitur(data_latih_numerik):
    probabilitas_fitur = {0: {}, 1: {}}
    for label in [0, 1]:  # 0 = Tidak, 1 = Ya
        data_label = [baris for baris in data_latih_numerik if baris[-1] == label]
        total_label = len(data_label)
        for indeks_fitur in range(len(data_latih_numerik[0]) - 1):
            jumlah_fitur = sum([baris[indeks_fitur] for baris in data_label])
            probabilitas_fitur[label][indeks_fitur] = (jumlah_fitur + 1) / (total_label + 2)  # Smoothing
    return probabilitas_fitur

# Fungsi klasifikasi
def klasifikasi(data, probabilitas_kelas, probabilitas_fitur):
    probabilitas_maks = -np.inf
    kelas_prediksi = None
    for label in probabilitas_kelas:
        probabilitas = log(probabilitas_kelas[label])
        for indeks_fitur, nilai_fitur in enumerate(data):
            probabilitas_f = probabilitas_fitur[label].get(indeks_fitur, 0.5)
            probabilitas += log(probabilitas_f if nilai_fitur == 1 else (1 - probabilitas_f))
        if probabilitas > probabilitas_maks:
            probabilitas_maks = probabilitas
            kelas_prediksi = label
    return kelas_prediksi

# Hitung probabilitas kelas dan fitur
probabilitas_kelas = hitung_probabilitas_kelas(data_latih_numerik)
probabilitas_fitur = hitung_probabilitas_fitur(data_latih_numerik)

# Prediksi data uji
prediksi = [klasifikasi(baris, probabilitas_kelas, probabilitas_fitur) for baris in data_uji_numerik]

# Tambahkan hasil prediksi ke data uji
for i, baris in enumerate(data_uji):
    baris.append('Ya' if prediksi[i] == 1 else 'Tidak')

# Tampilkan hasil
df = pd.DataFrame(data_uji, columns=['NIK', 'Nama', 'Lama Bekerja', 'Kualitas Bekerja', 'Perilaku', 'Absensi', 'Prediksi'])
print(df)
