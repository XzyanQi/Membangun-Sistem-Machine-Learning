# SMSML - Faqih Muhammad Ihsan

## Struktur Folder

- `Membangun_model/`
    - `modelling.py` : Script baseline model Logistic Regression.
    - `modelling_tuning.py` : Script model Logistic Regression + tuning hyperparameter.
    - `emails_preprocessed.csv` : Dataset hasil preprocessing (fitur numerik & label).
    - `screenshoot_dashboard.jpg` : SS MLflow UI lokal.
    - `screenshoot_artifak.jpg` : SS artefak model di MLflow lokal.
    - `requirements.txt` : List dependensi.
    - `DagsHub.txt` : Link repo DagsHub.

- `Workflow-CI.txt` : Catatan pipeline CI/CD pakai MLflow Project & Docker.

- `Monitoring dan Logging/`
    - `1.bukti_serving/` : SS serving model.
    - `2.prometheus.yml` : File config Prometheus (monitoring).
    - `3.prometheus_exporter.py` : Exporter Prometheus sederhana (custom metriks).
    - `4.bukti monitoring Prometheus/` : SS monitoring metriks di Prometheus.
    - `5.bukti monitoring Grafana/` : SS monitoring metriks di Grafana.
    - `6.bukti alerting Grafana/` : SS rules & notifikasi alert di Grafana.
    - `7.inference.py` : Script serving model FastAPI + metriks Prometheus.

## Cara Menjalankan

**Preprocessing**
- python automate_namafile.py
- File hasil preprocessing otomatis akan tersimpan di: preprocessing > namadataset_preprocessing > emails_preprocessed
- pakai dataset hasilnya

**Modelling & MLflow:**
- Pastikan sudah install dependensi di requirements.txt.
- Jalankan model baseline/tuning dengan:
1. python modelling.py
2. python modelling_tuning.py
3. mlflow ui lalu buka [http://127.0.0.1:5000](http://127.0.0.1:5000) di browser.

**Monitoring dan Logging**
- simpan prometheus.exe didekat prometheus.yml/.py
- uvicorn inference:app --host 0.0.0.0 --port 8000
- langsung ke url atau postman buat ujicoba
- ujicobanya pakai yang dataset hasil preprocessing, copy yang angka dan delete ,0 (nomor terakhir)
- selesai
