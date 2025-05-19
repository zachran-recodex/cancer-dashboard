from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer  # Import imputer untuk menangani missing values
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load preprocessed data
df_prep = pd.read_csv('data/preprocessed_cancer_data.csv')

# Select features for clustering
features = ['Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 'CancerStage_encoded', 
            'SmokingStatus_encoded', 'AlcoholUse_encoded', 'Gender_Male', 'Metastasis_Yes']

# Cek missing values sebelum preprocessing
print("Missing values in each feature:")
print(df_prep[features].isnull().sum())

# Ambil subset data yang berisi kolom-kolom yang diperlukan
X_raw = df_prep[features].values

# Impute missing values (ganti NaN dengan nilai mean)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_raw)

# Standardize features setelah imputasi
scaler = StandardScaler()
X = scaler.fit_transform(X_imputed)

# Cek apakah masih ada NaN values
print("NaN values after imputation:", np.isnan(X).sum())

# Determine optimal number of clusters using silhouette score
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score for Different k Values')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.savefig('image/silhouette_score.png')

# Implement K-means with optimal k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)
df_prep['Cluster'] = cluster_labels

# Visualize clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_prep['Cluster'], cmap='viridis', alpha=0.7)
plt.title('Visualization of Clusters in 2D PCA Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster')
plt.savefig('image/cluster_visualization.png')

# Analyze cluster characteristics
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

print(cluster_analysis)

# Menyimpan model dan scaler untuk digunakan di dashboard
import joblib
joblib.dump(kmeans, 'model/kmeans_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')