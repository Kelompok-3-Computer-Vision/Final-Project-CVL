# Final-Project-CVL

# Identifikasi Kondisi Kantuk Pengemudi Menggunakan MobileNetV2 dan XGBoost
Repositori ini berisi kode dan eksperimen untuk mendeteksi kondisi kantuk pengemudi (Drowsy vs Non-Drowsy) menggunakan karakteristik visual citra wajah dari Driver Drowsiness Dataset (DDD) di lingkungan Kaggle Notebook.

# Masalah & Tujuan
Masalah
Kondisi kantuk pengemudi secara signifikan memengaruhi kewaspadaan dan keselamatan berkendara di jalan raya. Karakteristik visual wajah, seperti kondisi mata dan ekspresi wajah, dapat dimanfaatkan untuk mengidentifikasi status pengemudi secara otomatis.

# Tujuan
Mengembangkan metode identifikasi kondisi kantuk pengemudi berdasarkan karakteristik visual citra wajah.
Memanfaatkan MobileNetV2 sebagai ekstraktor fitur visual yang efisien.
Menggunakan XGBoost sebagai model pengambilan keputusan (decision maker) berdasarkan representasi fitur dari MobileNetV2.
Mengevaluasi performa kombinasi hibrida MobileNetV2–XGBoost dibandingkan dengan pendekatan konvensional end-to-end CNN.

# 🧪 Desain Eksperimen
Eksperimen dibagi menjadi tiga skenario utama untuk menguji keunggulan pendekatan hibrida:
Metode
Arsitektur & Pendekatan
Peran / Keterangan
Baseline
MobileNetV2 + Softmax
CNN end-to-end dengan fine-tuning (25 layer terakhir dilatih)
Usulan
MobileNetV2 (frozen) + XGBoost
Pemisahan tegas antara ekstraksi fitur (feature extractor) dan pengambilan keputusan
Pembanding
ResNet50V2 + Softmax
Model CNN end-to-end alternatif sebagai pembanding arsitektur

Ketiga metode dievaluasi menggunakan metrik yang konsisten untuk memastikan perbandingan yang adil.

# ⚙️️ Persiapan Lingkungan (Kaggle)
Notebook ini dirancang khusus untuk dijalankan di platform Kaggle Notebook tanpa memerlukan Google Drive atau pengunduhan manual tambahan.
Aktifkan GPU: Di panel sebelah kanan Kaggle, buka Settings  Accelerator, lalu pilih GPU T4 x2 (atau P100).
Aktifkan Internet: Pastikan koneksi Internet aktif pada pengaturan untuk mengunduh bobot prarilis ImageNet.
Tambahkan Dataset: Klik tombol Add Data di panel kanan, lalu cari dan tambahkan Driver Drowsiness Dataset (DDD) oleh ismailnasri20.
Struktur Data:
Dataset dibaca langsung dari /kaggle/input/... secara read-only.
Proses splitting data (train, test, val) otomatis membuat symlink ke /kaggle/working/splitted_Data untuk efisiensi penyimpanan dan kecepatan akses disk.

# 📊 Metrik Evaluasi
Seluruh model dievaluasi berdasarkan metrik performa dan efisiensi komputasi berikut:
Accuracy, Precision, Recall, dan F1-Score
Confusion Matrix
Training Time (waktu pelatihan model)
Inference Time (waktu inferensi total dan rata-rata per citra)
