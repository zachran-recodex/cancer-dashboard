import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Buat direktori jika belum ada
if not os.path.exists('model'):
    os.makedirs('model')

print("Loading preprocessed data...")
df_prep = pd.read_csv('data/preprocessed_cancer_data.csv')
print(f"Dataset loaded with {df_prep.shape[0]} rows and {df_prep.shape[1]} columns")

# 1. KMEANS MODEL
print("\n--- Training KMeans Model ---")

# Select features for clustering
features = ['Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 'CancerStage_encoded', 
            'SmokingStatus_encoded', 'AlcoholUse_encoded', 'Gender_Male', 'Metastasis_Yes']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(df_prep[features])

# Save scaler
print("Saving scaler...")
joblib.dump(scaler, 'model/scaler.pkl')

# Determine optimal number of clusters using silhouette score
silhouette_scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg:.4f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Silhouette Score for Different k Values')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.savefig('model/silhouette_scores.png')

# Choose k=3 as determined in kmeans_clustering.py
optimal_k = 3
print(f"\nTraining final KMeans model with k={optimal_k}...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_prep['Cluster'] = kmeans.fit_predict(X)

# Save KMeans model
print("Saving KMeans model...")
joblib.dump(kmeans, 'model/kmeans_model.pkl')

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

print("\nCluster characteristics:")
print(cluster_analysis)

# Save the dataset with cluster assignments
df_prep.to_csv('data/preprocessed_cancer_data.csv', index=False)

# 2. LOGISTIC REGRESSION MODEL
print("\n--- Training Logistic Regression Model ---")

# Define features and target
X_all = df_prep[features]
y = df_prep['SurvivalStatus_Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Train logistic regression model with balanced class weights
log_model = LogisticRegression(class_weight='balanced', 
                               C=1.0,             # Regularization strength
                               penalty='l2',      # L2 regularization
                               solver='liblinear',# Solver algorithm
                               max_iter=1000,     # Maximum iterations
                               random_state=42)
log_model.fit(X_train, y_train)

# Evaluate model
y_pred = log_model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Logistic Regression Accuracy: {accuracy:.4f}")

# Save logistic regression model
print("Saving Logistic Regression model...")
joblib.dump(log_model, 'model/logistic_model.pkl')

print("\nAll models saved successfully!")