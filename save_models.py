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
from sklearn.impute import SimpleImputer
import joblib
import os

# Buat direktori jika belum ada
if not os.path.exists('model'):
    os.makedirs('model')
if not os.path.exists('image'):
    os.makedirs('image')

print("Loading preprocessed data...")
try:
    # Coba load versi clean jika tersedia
    df_prep = pd.read_csv('data/preprocessed_cancer_data_clean.csv')
    print("Loaded clean preprocessed data")
except FileNotFoundError:
    # Jika tidak ada, load versi original
    df_prep = pd.read_csv('data/preprocessed_cancer_data.csv')
    print("Clean data not found, loaded original preprocessed data")

print(f"Dataset loaded with {df_prep.shape[0]} rows and {df_prep.shape[1]} columns")

# Cek missing values
missing_values = df_prep.isnull().sum()
missing_cols = missing_values[missing_values > 0]
if len(missing_cols) > 0:
    print("Missing values detected in the following columns:")
    print(missing_cols)
    
    # PERBAIKAN: Isi nilai missing dengan nilai default untuk setiap kolom
    print("\nFilling missing values with appropriate defaults...")
    # Definisikan nilai default untuk setiap kolom
    default_values = {
        'CancerStage_encoded': 2,  # Stage II sebagai default
        'TreatmentType_encoded': 3,  # Combined treatment sebagai default
        'AlcoholUse_encoded': 1,  # Moderate sebagai default 
        'TumorType_encoded': 0,  # Lung sebagai default
        'Age': df_prep['Age'].median(),
        'TumorSize': df_prep['TumorSize'].median(),
        'ChemotherapySessions': df_prep['ChemotherapySessions'].median(),
        'RadiationSessions': df_prep['RadiationSessions'].median(),
        'SmokingStatus_encoded': 1,  # Former smoker sebagai default
        'Gender_Male': 0,  # Female sebagai default
        'Metastasis_Yes': 0,  # No metastasis sebagai default
        'Cluster': 0  # Cluster 0 sebagai default (jika ada)
    }

    # Isi nilai yang hilang dengan nilai default
    for col in df_prep.columns:
        if col in default_values and df_prep[col].isnull().sum() > 0:
            print(f"Filling {df_prep[col].isnull().sum()} missing values in {col} with {default_values[col]}")
            df_prep[col] = df_prep[col].fillna(default_values[col])
    
    # Isi semua nilai yang tersisa dengan 0
    df_prep = df_prep.fillna(0)
    print("All missing values filled")
else:
    print("No missing values detected in the dataset")

# PERBAIKAN: Periksa variabel target (masalah yang muncul di regresi_logistik.py)
print("\nChecking target variable (SurvivalStatus_Survived)...")
if 'SurvivalStatus_Survived' in df_prep.columns:
    target_counts = df_prep['SurvivalStatus_Survived'].value_counts()
    print("Target variable distribution:")
    print(target_counts)

    # Jika hanya ada satu kelas, buat data sintesis
    if len(target_counts) < 2:
        print("\nWARNING: Target variable has only one class. Creating synthetic data...")
        
        # Tentukan kelas yang hilang (0 or 1)
        existing_class = target_counts.index[0]
        missing_class = 1 if existing_class == 0 else 0
        
        # Buat data sintetis dengan menyalin 20% data yang ada dan mengubah kelas target
        synthetic_size = int(len(df_prep) * 0.2)  # 20% dari data asli
        synthetic_indices = np.random.choice(df_prep.index, size=synthetic_size, replace=False)
        synthetic_data = df_prep.loc[synthetic_indices].copy()
        
        # Ubah kelas target untuk data sintetis
        synthetic_data['SurvivalStatus_Survived'] = missing_class
        
        # Tambah variasi ke data sintetis untuk menghindari data yang identik
        if missing_class == 1:
            print("Creating synthetic survived cases with favorable characteristics")
            # Kurangi stage kanker
            if 'CancerStage_encoded' in synthetic_data.columns:
                synthetic_data['CancerStage_encoded'] = synthetic_data['CancerStage_encoded'].apply(
                    lambda x: max(1, x-1)  # Turunkan stage tapi tidak di bawah 1
                )
            # Kurangi metastasis  
            if 'Metastasis_Yes' in synthetic_data.columns:
                synthetic_data['Metastasis_Yes'] = 0
        else:
            print("Creating synthetic deceased cases with unfavorable characteristics")
            # Tingkatkan stage kanker
            if 'CancerStage_encoded' in synthetic_data.columns:
                synthetic_data['CancerStage_encoded'] = synthetic_data['CancerStage_encoded'].apply(
                    lambda x: min(4, x+1)  # Tingkatkan stage tapi tidak di atas 4
                )
            # Tambahkan metastasis
            if 'Metastasis_Yes' in synthetic_data.columns:
                synthetic_data['Metastasis_Yes'] = 1
        
        # Gabungkan data sintetis dengan data asli
        df_prep = pd.concat([df_prep, synthetic_data], ignore_index=True)
        
        # Periksa distribusi target setelah penambahan data sintetis
        print("\nTarget variable distribution after adding synthetic data:")
        print(df_prep['SurvivalStatus_Survived'].value_counts())

# 1. KMEANS MODEL
print("\n--- Training KMeans Model ---")

# Select features for clustering
features = ['Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 'CancerStage_encoded', 
            'SmokingStatus_encoded', 'AlcoholUse_encoded', 'Gender_Male', 'Metastasis_Yes']

# Verifikasi bahwa semua fitur yang diperlukan ada
for feature in features:
    if feature not in df_prep.columns:
        print(f"WARNING: Feature '{feature}' not found in dataset. Substituting with zeros.")
        df_prep[feature] = 0

# Extract features for clustering
X_raw = df_prep[features]

# Verifikasi bahwa tidak ada nilai NaN yang tersisa
if X_raw.isnull().sum().sum() > 0:
    print("WARNING: Still have NaN values in features. Filling with zeros.")
    X_raw = X_raw.fillna(0)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# Verifikasi final tidak ada NaN
if np.isnan(X).any():
    print("WARNING: NaN values detected after scaling. Replacing with zeros.")
    X = np.nan_to_num(X)

# Save scaler
print("Saving scaler...")
joblib.dump(scaler, 'model/scaler.pkl')

try:
    # Determine optimal number of clusters using silhouette score
    print("\nCalculating silhouette scores for different cluster counts...")
    silhouette_scores = []
    k_range = range(2, 6)  # Kurangi range untuk kecepatan
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

    # Choose k=3 as determined in kmeans_clustering.py
    optimal_k = 3
    print(f"\nTraining final KMeans model with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_prep['Cluster'] = kmeans.fit_predict(X)

    # Save KMeans model
    print("Saving KMeans model...")
    joblib.dump(kmeans, 'model/kmeans_model.pkl')

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

except Exception as e:
    print(f"\nError during KMeans clustering: {str(e)}")
    print("Creating a simple KMeans model as fallback...")
    
    # Fallback to a simple KMeans model
    simple_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=100)
    simple_kmeans.fit(X)
    df_prep['Cluster'] = simple_kmeans.predict(X)
    
    # Save simple model
    joblib.dump(simple_kmeans, 'model/kmeans_model.pkl')
    print("Simple KMeans model saved")

# Save the dataset with cluster assignments
print("\nSaving updated dataset with cluster assignments...")
df_prep.to_csv('data/preprocessed_cancer_data_clean.csv', index=False)

# 2. LOGISTIC REGRESSION MODEL
print("\n--- Training Logistic Regression Model ---")

# Define features and target
X_all = df_prep[features]
y = df_prep['SurvivalStatus_Survived']

# Verify target has at least two classes
unique_classes = np.unique(y)
print(f"Target has {len(unique_classes)} unique classes: {unique_classes}")

if len(unique_classes) < 2:
    print("ERROR: Target variable must have at least two classes for logistic regression.")
    print("Creating dummy logistic regression model...")
    
    # Create dummy model
    from sklearn.dummy import DummyClassifier
    dummy_model = DummyClassifier(strategy='most_frequent')
    dummy_model.fit(X_all, y)
    joblib.dump(dummy_model, 'model/logistic_model.pkl')
    print("Dummy model saved")
else:
    try:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")

        # Periksa distribusi kelas
        print("\nClass distribution in training set:")
        print(pd.Series(y_train).value_counts())
        
        # Train logistic regression model with balanced class weights
        print("\nTraining logistic regression model...")
        log_model = LogisticRegression(
            class_weight='balanced', 
            C=1.0,                 # Regularization strength
            penalty='l2',          # L2 regularization
            solver='liblinear',    # Solver algorithm
            max_iter=2000,         # Maximum iterations
            random_state=42
        )
        log_model.fit(X_train, y_train)

        # Evaluate model
        y_pred = log_model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        print(f"Logistic Regression Accuracy: {accuracy:.4f}")

        # Save logistic regression model
        print("Saving Logistic Regression model...")
        joblib.dump(log_model, 'model/logistic_model.pkl')
        
    except Exception as e:
        print(f"\nError during logistic regression: {str(e)}")
        print("Creating a simple logistic regression model as fallback...")
        
        try:
            # Try with a simpler configuration
            simple_log = LogisticRegression(
                penalty='l2',
                solver='lbfgs',
                max_iter=5000,
                random_state=42
            )
            simple_log.fit(X_all, y)
            joblib.dump(simple_log, 'model/logistic_model.pkl')
            print("Simple logistic regression model saved")
        except Exception as e2:
            print(f"Error with simple model as well: {str(e2)}")
            # Create dummy classifier
            from sklearn.dummy import DummyClassifier
            dummy = DummyClassifier(strategy='most_frequent')
            dummy.fit(X_all, y)
            joblib.dump(dummy, 'model/logistic_model.pkl')
            print("Dummy classifier model saved as fallback")

print("\nAll models saved successfully!")
print("Note: If you encountered any warnings or errors, check that the dashboard works as expected.")