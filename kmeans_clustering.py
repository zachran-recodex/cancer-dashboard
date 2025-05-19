from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Pastikan direktori image ada
if not os.path.exists('image'):
    os.makedirs('image')

print("Loading preprocessed data...")
# Load preprocessed data
df_prep = pd.read_csv('data/preprocessed_cancer_data.csv')
print(f"Dataset loaded with {df_prep.shape[0]} rows and {df_prep.shape[1]} columns")

# Cek missing values
missing_values = df_prep.isnull().sum()
missing_cols = missing_values[missing_values > 0]
if len(missing_cols) > 0:
    print("Missing values detected in the following columns:")
    print(missing_cols)
else:
    print("No missing values detected in the dataset")

# Select features for clustering
features = ['Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 'CancerStage_encoded', 
            'SmokingStatus_encoded', 'AlcoholUse_encoded', 'Gender_Male', 'Metastasis_Yes']

# Extract features for clustering
X_raw = df_prep[features]

# PERBAIKAN: Tangani missing values dengan SimpleImputer
print("\nHandling missing values with mean imputation...")
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X_raw)

# Cek apakah masih ada NaN setelah imputasi
if np.isnan(X).any():
    print("Warning: NaN values still present after imputation")
    # Jika masih ada NaN, ganti dengan 0
    X = np.nan_to_num(X)
else:
    print("All missing values have been successfully imputed")

# Standardize features
print("Standardizing features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Double-check untuk NaN setelah scaling
if np.isnan(X).any():
    print("Warning: NaN values introduced during scaling")
    X = np.nan_to_num(X)
else:
    print("No NaN values after standardization")

# Determine optimal number of clusters using silhouette score
print("\nCalculating silhouette scores for different cluster counts...")
silhouette_scores = []
k_range = range(2, 11)
for k in k_range:
    print(f"Testing k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg:.4f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(list(k_range), silhouette_scores, marker='o')
plt.title('Silhouette Score for Different k Values')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('image/silhouette_score.png')
print("Silhouette score plot saved")

# Implement K-means with optimal k=3
print("\nTraining final K-means model with k=3...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

# Add cluster labels to dataframe
df_prep['Cluster'] = cluster_labels

# Visualize clusters using PCA
print("Visualizing clusters with PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_prep['Cluster'], cmap='viridis', alpha=0.7)
plt.title('Visualization of Clusters in 2D PCA Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig('image/cluster_visualization.png')
print("Cluster visualization saved")

# Analyze cluster characteristics
print("\nAnalyzing cluster characteristics...")
cluster_analysis = df_prep.groupby('Cluster').agg({
    'Age': 'mean',
    'TumorSize': 'mean',
    'ChemotherapySessions': 'mean',
    'RadiationSessions': 'mean',
    'CancerStage_encoded': 'mean',
    'SmokingStatus_encoded': 'mean',
    'AlcoholUse_encoded': 'mean',
    'Gender_Male': 'mean',
    'Metastasis_Yes': 'mean',
    'SurvivalStatus_Survived': 'mean'
})

print("\nCluster characteristics:")
print(cluster_analysis)

# Save the updated dataframe with cluster assignments
print("\nSaving preprocessed data with cluster assignments...")
df_prep.to_csv('data/preprocessed_cancer_data.csv', index=False)
print("Clustering process completed successfully!")