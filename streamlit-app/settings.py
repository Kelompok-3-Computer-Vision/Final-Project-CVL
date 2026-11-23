"""Tunable constants for the live webcam pipeline (kept free of heavy imports)."""

DEFAULT_MAX_INFERENCE_FPS = 5.0   # upper limit on analysed frames per second
STALE_RESULT_SECONDS = 2.0        # a result older than this is no longer drawn / shown as current
FPS_WINDOW = 30                   # frames / inferences used for rolling averages

# Temporal smoothing: jumlah frame terakhir untuk rolling average probabilitas
SMOOTHING_WINDOW = 5              # semakin besar = semakin stabil, tapi lebih lambat bereaksi

# Hysteresis threshold: hindari flipping di sekitar 0.5
HYSTERESIS_HIGH = 0.60            # prob_drowsy >= nilai ini → pindah ke status Drowsy
HYSTERESIS_LOW  = 0.38            # prob_drowsy <= nilai ini → pindah ke status Non Drowsy
# Di antara HYSTERESIS_LOW dan HYSTERESIS_HIGH → pertahankan status sebelumnya

# Haar Cascade tuning
HAAR_SCALE_FACTOR  = 1.05         # lebih halus dari default 1.1 → deteksi lebih stabil
HAAR_MIN_NEIGHBORS = 6            # lebih ketat dari default 5 → lebih sedikit false positive
