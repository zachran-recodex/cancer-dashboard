import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data/china_cancer_patients_synthetic.csv')

# Display basic information
print(df.info())
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Exploratory Data Analysis
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True)
plt.title('Distribusi Usia Pasien Kanker')
plt.xlabel('Usia')
plt.ylabel('Frekuensi')
plt.savefig('image/age_distribution.png')

# Cross-tabulation
tumor_gender = pd.crosstab(df['TumorType'], df['Gender'])
tumor_gender.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Distribusi Jenis Kanker Berdasarkan Gender')
plt.xlabel('Jenis Kanker')
plt.ylabel('Jumlah Pasien')
plt.savefig('image/cancer_type_by_gender.png')

# Correlation analysis
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Korelasi antar Variabel Numerik')
plt.savefig('image/correlation_heatmap.png')