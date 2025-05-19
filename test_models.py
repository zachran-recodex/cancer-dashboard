import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import joblib
import os

# Buat direktori untuk menyimpan hasil evaluasi
if not os.path.exists('results'):
    os.makedirs('results')

print("Loading data and models...")
# Load preprocessed data
df_prep = pd.read_csv('data/preprocessed_cancer_data.csv')

# Load models
kmeans_model = joblib.load('model/kmeans_model.pkl')
logistic_model = joblib.load('model/logistic_model.pkl')
scaler = joblib.load('model/scaler.pkl')

print(f"Dataset loaded with {df_prep.shape[0]} rows and {df_prep.shape[1]} columns")

# 1. TEST KMEANS MODEL
print("\n--- Testing KMeans Model ---")

# Features used for clustering
features = ['Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 'CancerStage_encoded', 
            'SmokingStatus_encoded', 'AlcoholUse_encoded', 'Gender_Male', 'Metastasis_Yes']

# Scale features
X = scaler.transform(df_prep[features])

# Predict clusters
df_prep['PredictedCluster'] = kmeans_model.predict(X)

# Compare original clusters with predicted clusters
cluster_match = (df_prep['Cluster'] == df_prep['PredictedCluster']).mean() * 100
print(f"Cluster prediction accuracy: {cluster_match:.2f}%")

# Analyze cluster profiles
cluster_profiles = df_prep.groupby('PredictedCluster').agg({
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

print("\nCluster Profiles:")
print(cluster_profiles)

# Visualize cluster distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='PredictedCluster', data=df_prep)
plt.title('Distribution of Patients Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.savefig('results/cluster_distribution.png')

# Visualize survival rate by cluster
plt.figure(figsize=(10, 6))
survival_by_cluster = df_prep.groupby('PredictedCluster')['SurvivalStatus_Survived'].mean()
sns.barplot(x=survival_by_cluster.index, y=survival_by_cluster.values)
plt.title('Survival Rate by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Survival Rate')
plt.ylim(0, 1)
plt.savefig('results/survival_by_cluster.png')

# 2. TEST LOGISTIC REGRESSION MODEL
print("\n--- Testing Logistic Regression Model ---")

# Define features and target for test data
X_test = df_prep[features]
y_test = df_prep['SurvivalStatus_Survived']

# Scale test data
X_test_scaled = scaler.transform(X_test)

# Make predictions
y_pred = logistic_model.predict(X_test_scaled)
y_pred_proba = logistic_model.predict_proba(X_test_scaled)[:, 1]

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('results/confusion_matrix.png')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('results/roc_curve.png')

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': logistic_model.coef_[0],
    'Odds_Ratio': np.exp(logistic_model.coef_[0])
})
feature_importance['Absolute_Coefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Absolute_Coefficient', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(12, 8))
plt.barh(feature_importance['Feature'], feature_importance['Absolute_Coefficient'])
plt.title('Feature Importance')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('results/feature_importance.png')

# 3. SIMULATION - Test with sample patient
print("\n--- Testing with a sample patient ---")

# Create a sample patient
sample_patient = {
    'Age': 65,
    'TumorSize': 3.5,
    'ChemotherapySessions': 6,
    'RadiationSessions': 12,
    'CancerStage_encoded': 3,
    'SmokingStatus_encoded': 2,
    'AlcoholUse_encoded': 1,
    'Gender_Male': 1,
    'Metastasis_Yes': 0
}

# Convert to DataFrame
sample_df = pd.DataFrame([sample_patient])

# Scale the features
sample_scaled = scaler.transform(sample_df)

# Predict cluster
sample_cluster = kmeans_model.predict(sample_scaled)[0]
print(f"Predicted Cluster: {sample_cluster}")

# Predict survival
survival_proba = logistic_model.predict_proba(sample_scaled)[0, 1]
survival_pred = "Survived" if survival_proba > 0.5 else "Deceased"
print(f"Survival Prediction: {survival_pred} (Probability: {survival_proba:.4f})")

print("\nTesting completed successfully!")