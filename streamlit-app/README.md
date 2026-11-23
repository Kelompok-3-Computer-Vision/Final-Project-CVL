# 🚗 Driver Drowsiness Detection System

Sistem pemantau kondisi pengemudi (*Driver Drowsiness Monitoring*) berbasis web menggunakan kombinasi **MobileNetV2 (Frozen Feature Extractor)** dan **XGBoost Classifier**, dilengkapi dengan **Temporal Smoothing** dan **Hysteresis Thresholding** untuk mencegah prediksi berkedip/bolak-balik (*flipping*).

---

## 🌟 Fitur Utama
1. **📷 Realtime Webcam**: Deteksi kantuk secara langsung melalui kamera laptop/webcam dengan WebRTC stream yang stabil, visual bounding box, FPS, dan metrik latensi per-tahap.
2. **🎞️ Video Analysis**: Unggah rekaman video mengemudi untuk menganalisis frekuensi kantuk, grafik probabilitas kantuk per frame, dan statistik total.
3. **🖼️ Image Analysis**: Unggah foto pengemudi untuk mengidentifikasi kondisi (*Drowsy* vs *Non-Drowsy*) beserta probabilitas masing-masing kelas.
4. **📊 Overview & About**: Penjelasan arsitektur sistem, dataset, metrik evaluasi, dan pipeline model.

---

## 🏗️ Arsitektur Model & Pipeline
```
Input Frame (Kamera / Video / Foto)
           │
           ▼
[1] Face Detection (Haar Cascade Classifier - Stabilized)
           │
           ▼
[2] Crop Face & Preprocessing (Resize 160x160, Normalization /255.0)
           │
           ▼
[3] Feature Extraction: MobileNetV2 (Pre-trained, Frozen)
           │  (Vector Fitur)
           ▼
[4] Classification: XGBoost Classifier
           │  (Raw Probabilities)
           ▼
[5] Temporal Smoothing & Hysteresis
    - Rolling Average Buffer (5 frame)
    - Threshold Masuk Drowsy: >= 0.60
    - Threshold Keluar Drowsy: <= 0.38
           │
           ▼
Hasil Prediksi Stabil (Drowsy / Non-Drowsy)
```

---

## 📁 Struktur Direktori
```text
driver-drowsiness-detection/
├── .streamlit/
│   └── config.toml                  # Konfigurasi tema UI Streamlit
├── views/
│   ├── __init__.py
│   ├── about.py                     # Halaman About
│   ├── image_analysis.py            # Halaman analisis foto
│   ├── live_webcam.py               # Halaman webcam real-time
│   ├── overview.py                  # Halaman dashboard overview
│   └── video_analysis.py            # Halaman analisis video
├── app.py                           # File utama (entry point) aplikasi
├── model_utils.py                   # Fungsi load model & inferensi
├── webcam_processor.py              # Handler video stream WebRTC & smoothing worker
├── overlay.py                       # Visual annotator (bounding box & label)
├── settings.py                      # Konstanta konfigurasi & tuning model
├── theme.py                         # Styling CSS komponen antarmuka
├── nav.py                           # Routing navigasi halaman
├── requirements.txt                 # Daftar dependencies Python
├── .gitignore                       # File aturan ignore Git
├── metadata.json                    # Metadata mapping label dan ukuran input
├── mobilenetv2_feature_extractor.keras  # Bobot model MobileNetV2
└── xgb_drowsiness_model.json        # Model XGBoost Classifier
```

---

## 🚀 Panduan Instalasi & Menjalankan

### 1. Prasyarat
- Python 3.10 atau Python 3.11 (Sangat disarankan **Python 3.11** untuk stabilitas TensorFlow & OpenCV).
- Git (jika melakukan clone repository).

### 2. Setup Virtual Environment
```bash
# Clone repository
git clone https://github.com/USERNAME/driver-drowsiness-detection.git
cd driver-drowsiness-detection

# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependensi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
```bash
streamlit run app.py
```
Aplikasi akan otomatis berjalan di browser pada alamat:
👉 **`http://localhost:8501`**

---

## ⚙️ Penyesuaian Sensitivitas Deteksi
Anda dapat mengubah nilai parameter stabilisasi pada file `settings.py`:
- `DEFAULT_MAX_INFERENCE_FPS`: Batas inferensi per detik (default: `5.0`).
- `SMOOTHING_WINDOW`: Jumlah frame riwayat untuk rata-rata probabilitas (default: `5`).
- `HYSTERESIS_HIGH`: Batas minimal probabilitas untuk beralih ke status **Drowsy** (default: `0.60`).
- `HYSTERESIS_LOW`: Batas maksimal probabilitas untuk kembali ke status **Non-Drowsy** (default: `0.38`).

---

## 📄 Lisensi & Kontribusi
Dikembangkan untuk keperluan penelitian dan tugas Computer Vision / Machine Learning.
Silakan buat Pull Request atau ajukan Issue jika ada pertanyaan maupun saran perbaikan.
