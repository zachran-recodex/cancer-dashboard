# Analisis dan Prediksi Pasien Kanker

Proyek ini menyediakan analisis komprehensif dan prediksi kemungkinan kelangsungan hidup pasien kanker menggunakan teknik machine learning. Proyek diimplementasikan dalam Python dan menggunakan berbagai library data science seperti pandas, scikit-learn, dan Dash untuk visualisasi interaktif.

## Struktur Proyek

```
project/
│
├── data/
│   ├── china_cancer_patients_synthetic.csv       # Dataset original
│   └── preprocessed_cancer_data.csv              # Dataset yang sudah diproses
│
├── model/
│   ├── kmeans_model.pkl                          # Model K-means
│   ├── logistic_model.pkl                        # Model regresi logistik
│   └── scaler.pkl                                # Scaler untuk normalisasi data
│
├── image/                                        # Output visualisasi
│
├── results/                                      # Hasil evaluasi model
│
├── eksplorasi_data.py                            # Eksplorasi data awal
├── preprocessing.py                              # Preprocessing data
├── kmeans_clustering.py                          # Analisis clustering
├── regresi_logistik.py                           # Model prediksi survival
├── save_models.py                                # Menyimpan model terlatih
├── test_models.py                                # Pengujian model
├── config.py                                     # Konfigurasi untuk dashboard
└── app.py                                        # Dashboard interaktif
```

## Setup dan Instalasi

1. Clone repositori ini
2. Buat virtual environment (opsional)
3. Instal dependensi:

```bash
pip install -r requirements.txt
```

## Alur Kerja Analisis

1. **Eksplorasi Data**: Menjalankan analisis awal dataset pasien kanker.
   ```bash
   python eksplorasi_data.py
   ```

2. **Preprocessing Data**: Memproses data mentah untuk analisis lanjutan.
   ```bash
   python preprocessing.py
   ```

3. **Clustering**: Mengelompokkan pasien berdasarkan karakteristik serupa.
   ```bash
   python kmeans_clustering.py
   ```

4. **Pemodelan Prediktif**: Membangun model untuk memprediksi kelangsungan hidup.
   ```bash
   python regresi_logistik.py
   ```

5. **Menyimpan Model**: Menyimpan model terlatih untuk digunakan di dashboard.
   ```bash
   python save_models.py
   ```

6. **Pengujian Model**: Mengevaluasi performa model.
   ```bash
   python test_models.py
   ```

7. **Dashboard Interaktif**: Menjalankan aplikasi web untuk visualisasi dan prediksi.
   ```bash
   python app.py
   ```

## Dataset

Dataset berisi data sintetis 10.000 pasien kanker dari China dengan 20 variabel, termasuk:
- Informasi demografis (usia, jenis kelamin, dsb.)
- Detail klinis (jenis kanker, stadium, ukuran tumor)
- Informasi pengobatan (kemoterapi, radiasi)
- Faktor risiko (merokok, alkohol)
- Status kelangsungan hidup

## Hasil Analisis

### Clustering
Model K-means mengidentifikasi 3 cluster pasien:
- **Cluster 0**: "Young Survivors" - Pasien muda dengan prognosis baik
- **Cluster 1**: "Mid-stage Patients" - Pasien dengan stadium menengah
- **Cluster 2**: "Advanced Cases" - Kasus lanjut dengan prognosis kurang baik

### Model Prediktif
Model regresi logistik digunakan untuk memprediksi status kelangsungan hidup dengan performa:
- Accuracy: ~0.85
- Precision: ~0.83
- Recall: ~0.84
- F1-score: ~0.83
- AUC-ROC: ~0.89

### Faktor Penting
Faktor utama yang mempengaruhi kelangsungan hidup:
1. Stadium kanker
2. Metastasis
3. Ukuran tumor
4. Usia
5. Jumlah sesi kemoterapi

## Dashboard

Dashboard interaktif menyediakan:
- Visualisasi distribusi demografis
- Eksplorasi interaktif hubungan antar variabel
- Perbandingan karakteristik cluster
- Alat prediksi kelangsungan hidup
- Metrik performa model

Akses dashboard dengan menjalankan:
```bash
python app.py
```

## Dependensi

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- dash
- plotly
- joblib

## Kontribusi

Kontribusi dipersilakan! Silakan fork repositori ini dan ajukan pull request.

## Lisensi

Proyek ini dilisensikan di bawah lisensi MIT.