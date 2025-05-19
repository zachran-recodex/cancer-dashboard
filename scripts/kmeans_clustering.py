import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def perform_kmeans_clustering(file_path):
    """
    Perform K-means clustering on the preprocessed data.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the preprocessed data
    
    Returns:
    --------
    pd.DataFrame
        Data with cluster labels
    """
    print(f"Loading preprocessed data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Select features for clustering
    clustering_features = [
        'Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions',
        'CancerStage_encoded', 'SmokingStatus_encoded', 'AlcoholUse_encoded',
        'Gender_Male', 'Metastasis_Yes', 'TumorType_encoded'
    ]
    
    # Check for missing values
    missing_values = df[clustering_features].isnull().sum()
    print("\nMissing values in features:")
    print(missing_values)
    
    # Handle missing values using SimpleImputer
    print("Imputing missing values...")
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(df[clustering_features].values)
    
    # Determine optimal number of clusters using silhouette score
    print("Determining optimal number of clusters...")
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        print(f"Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"Silhouette score for k={k}: {silhouette_avg:.4f}")
    
    # Plotting silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Method For Optimal k')
    plt.grid(True)
    
    os.makedirs('../dashboard/assets', exist_ok=True)
    plt.savefig('../dashboard/assets/silhouette_scores.png')
    
    # Determine optimal k from silhouette scores
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
    
    # Perform K-means clustering with optimal k
    print(f"Performing K-means clustering with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Save the trained model
    os.makedirs('../dashboard/models', exist_ok=True)
    joblib.dump(kmeans, '../dashboard/models/kmeans_model.pkl')
    
    # Analyze clusters
    print("Analyzing clusters...")
    cluster_analysis = df.groupby('Cluster').agg({
        'Age': 'mean',
        'TumorSize': 'mean',
        'ChemotherapySessions': 'mean',
        'RadiationSessions': 'mean',
        'CancerStage_encoded': 'mean',
        'SmokingStatus_encoded': 'mean',
        'AlcoholUse_encoded': 'mean',
        'Gender_Male': 'mean',
        'Metastasis_Yes': 'mean',
        'SurvivalStatus_Survived': 'mean',
        'PatientID': 'count'
    }).rename(columns={'PatientID': 'Count'})
    
    print("\nCluster Analysis:")
    print(cluster_analysis)
    
    # Calculate percentage of patients in each cluster
    total_patients = df.shape[0]
    cluster_percentages = (cluster_analysis['Count'] / total_patients * 100).round(2)
    print("\nPercentage of patients in each cluster:")
    print(cluster_percentages)
    
    # Visualize clusters using PCA
    print("Visualizing clusters using PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create a dataframe for the PCA results
    pca_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': df['Cluster']
    })
    
    # Save PCA components for dashboard
    pca_df.to_csv('../dashboard/assets/pca_results.csv', index=False)
    
    # Plot clusters
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='viridis', s=50, alpha=0.7)
    plt.title('Visualization of Clusters using PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.savefig('../dashboard/assets/cluster_visualization.png')
    
    # Save the final dataset with cluster labels
    output_path = '../data/clustered_cancer_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Data with cluster labels saved to {output_path}")
    
    # Name the clusters based on their characteristics
    cluster_names = {}
    
    for cluster in range(optimal_k):
        cluster_data = df[df['Cluster'] == cluster]
        
        # Base characteristics
        avg_age = cluster_data['Age'].mean()
        avg_stage = cluster_data['CancerStage_encoded'].mean()
        avg_tumor_size = cluster_data['TumorSize'].mean()
        metastasis_rate = cluster_data['Metastasis_Yes'].mean()
        survival_rate = cluster_data['SurvivalStatus_Survived'].mean()
        
        # Determine cluster name based on key characteristics
        if avg_age < -0.5 and avg_stage < -0.4 and metastasis_rate < 0.2 and survival_rate > 0.8:
            name = "Young Survivors"
        elif avg_age > 0.5 and avg_stage > 0.5 and metastasis_rate > 0.7 and survival_rate < 0.3:
            name = "Advanced Cases"
        else:
            name = "Mid-stage Patients"
        
        cluster_names[cluster] = name
        
    print("\nCluster Names:")
    for cluster, name in cluster_names.items():
        print(f"Cluster {cluster}: {name}")
    
    # Save cluster names
    cluster_name_df = pd.DataFrame(list(cluster_names.items()), columns=['Cluster', 'Name'])
    cluster_name_df.to_csv('../dashboard/assets/cluster_names.csv', index=False)
    
    return df

if __name__ == "__main__":
    perform_kmeans_clustering('../data/preprocessed_cancer_data.csv')