"""
Konfigurasi untuk aplikasi dashboard kanker
File ini berisi definisi konstan dan konfigurasi
yang digunakan di seluruh aplikasi dashboard
"""

# Definisi warna untuk visualisasi
COLOR_PALETTE = {
    'primary': '#1F77B4',
    'secondary': '#FF7F0E',
    'tertiary': '#2CA02C',
    'quaternary': '#D62728',
    'background': '#F8F9FA',
    'text': '#212529',
    'cluster0': '#3366CC',
    'cluster1': '#DC3912',
    'cluster2': '#FF9900'
}

# Konfigurasi jumlah cluster
NUM_CLUSTERS = 3

# Deskripsi cluster
CLUSTER_DESCRIPTIONS = {
    0: "Young Survivors - Pasien yang lebih muda dengan stadium awal dan tingkat kelangsungan hidup tinggi",
    1: "Mid-stage Patients - Pasien dengan stadium menengah dan pengobatan kombinasi",
    2: "Advanced Cases - Kasus lanjut dengan metastasis dan prognosis yang kurang baik"
}

# Pemetaan variabel untuk dropdown
FEATURE_LABELS = {
    'Age': 'Usia',
    'TumorSize': 'Ukuran Tumor',
    'ChemotherapySessions': 'Sesi Kemoterapi',
    'RadiationSessions': 'Sesi Radiasi',
    'CancerStage_encoded': 'Stadium Kanker',
    'SmokingStatus_encoded': 'Status Merokok',
    'AlcoholUse_encoded': 'Konsumsi Alkohol',
    'Gender_Male': 'Jenis Kelamin (1=Pria)',
    'Metastasis_Yes': 'Metastasis (1=Ya)',
    'TumorType_encoded': 'Jenis Tumor',
    'SurvivalStatus_Survived': 'Status Kelangsungan Hidup',
    'FollowUpMonths': 'Bulan Follow-Up',
    'Cluster': 'Cluster'
}

# Definisi pemetaan untuk kategori
TUMOR_TYPE_MAPPING = {
    0: 'Paru-paru',
    1: 'Kolorektal',
    2: 'Payudara', 
    3: 'Hati',
    4: 'Lambung',
    5: 'Lainnya'
}

TREATMENT_TYPE_MAPPING = {
    0: 'Operasi',
    1: 'Kemoterapi',
    2: 'Radiasi',
    3: 'Kombinasi',
    4: 'Paliatif'
}

CANCER_STAGE_MAPPING = {
    1: 'Stadium I',
    2: 'Stadium II',
    3: 'Stadium III',
    4: 'Stadium IV'
}

SMOKING_STATUS_MAPPING = {
    0: 'Tidak Pernah',
    1: 'Mantan Perokok',
    2: 'Perokok Aktif'
}

ALCOHOL_USE_MAPPING = {
    0: 'Tidak',
    1: 'Moderat',
    2: 'Berat'
}

# Konfigurasi visualisasi model
MODEL_METRICS = {
    'accuracy': 'Accuracy (Akurasi)',
    'precision': 'Precision (Presisi)',
    'recall': 'Recall (Sensitivitas)',
    'f1': 'F1-Score',
    'auc': 'AUC-ROC'
}