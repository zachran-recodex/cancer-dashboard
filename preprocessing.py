import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Buat direktori jika belum ada
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('model'):
    os.makedirs('model')

# Load dataset original
print("Loading original dataset...")
df = pd.read_csv('data/china_cancer_patients_synthetic.csv')
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# Periksa missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values[missing_values > 0])

# Tangani missing values
print("\nHandling missing values...")
# Isi missing values numerik dengan mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

# Isi missing values kategorikal dengan mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Encoding untuk variabel kategorikal
print("\nEncoding categorical variables...")
# Encoding untuk TumorType
tumor_type_mapping = {
    'Lung': 0,
    'Colorectal': 1,
    'Breast': 2,
    'Liver': 3,
    'Stomach': 4,
    'Other': 5
}
df['TumorType_encoded'] = df['TumorType'].map(tumor_type_mapping)

# Encoding untuk TreatmentType
treatment_type_mapping = {
    'Surgery': 0,
    'Chemotherapy': 1,
    'Radiation': 2,
    'Combined': 3,
    'Palliative': 4
}
df['TreatmentType_encoded'] = df['TreatmentType'].map(treatment_type_mapping)

# Encoding untuk CancerStage
cancer_stage_mapping = {
    'Stage I': 1,
    'Stage II': 2,
    'Stage III': 3,
    'Stage IV': 4
}
df['CancerStage_encoded'] = df['CancerStage'].map(cancer_stage_mapping)

# Encoding untuk kolom biner
df['Gender_Male'] = (df['Gender'] == 'Male').astype(int)
df['Metastasis_Yes'] = (df['Metastasis'] == 'Yes').astype(int)
df['SurvivalStatus_Survived'] = (df['SurvivalStatus'] == 'Survived').astype(int)

# Label encoding untuk SmokingStatus dan AlcoholUse
smoking_status_mapping = {
    'Never': 0,
    'Former': 1,
    'Current': 2
}
df['SmokingStatus_encoded'] = df['SmokingStatus'].map(smoking_status_mapping)

alcohol_use_mapping = {
    'None': 0,
    'Moderate': 1,
    'Heavy': 2
}
df['AlcoholUse_encoded'] = df['AlcoholUse'].map(alcohol_use_mapping)

# Seleksi fitur untuk dataset final
preprocessed_cols = [
    'PatientID', 'Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions',
    'CancerStage_encoded', 'TreatmentType_encoded', 'SmokingStatus_encoded', 
    'AlcoholUse_encoded', 'Gender_Male', 'Metastasis_Yes', 'TumorType_encoded',
    'SurvivalStatus_Survived', 'FollowUpMonths'
]

# Buat dataset yang preprocessed
preprocessed_df = df[preprocessed_cols]

# Simpan dataset preprocessed
preprocessed_df.to_csv('data/preprocessed_cancer_data.csv', index=False)
print(f"\nPreprocessed dataset saved with {preprocessed_df.shape[0]} rows and {preprocessed_df.shape[1]} columns")
print("Preprocessing completed successfully!")