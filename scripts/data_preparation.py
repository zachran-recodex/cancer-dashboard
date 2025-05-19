import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import datetime

def preprocess_data(file_path):
    """
    Preprocess the cancer patient dataset for analysis.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the cancer patient data
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed data ready for clustering and classification
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Original dataset shape: {df.shape}")
    
    # 1. Handle missing values
    print("Handling missing values...")
    
    # Check missing values
    missing_values = df.isnull().sum()
    print("Missing values per column:")
    print(missing_values[missing_values > 0])
    
    # Fill missing values
    # For TumorSize, fill with median
    df['TumorSize'].fillna(df['TumorSize'].median(), inplace=True)
    
    # For categorical variables, fill with mode
    df['GeneticMutation'].fillna('Unknown', inplace=True)
    df['Comorbidities'].fillna('None', inplace=True)
    
    # 2. Feature engineering
    print("Performing feature engineering...")
    
    # Convert dates to datetime
    df['DiagnosisDate'] = pd.to_datetime(df['DiagnosisDate'])
    df['SurgeryDate'] = pd.to_datetime(df['SurgeryDate'])
    
    # Calculate waiting time for surgery (days)
    df['WaitingTimeDays'] = (df['SurgeryDate'] - df['DiagnosisDate']).dt.days
    
    # Fill NaN in WaitingTimeDays (for patients without surgery)
    df['WaitingTimeDays'].fillna(0, inplace=True)
    
    # Create treatment complexity variable
    def get_treatment_complexity(row):
        if row['TreatmentType'] == 'Palliative':
            return 'Low'
        elif row['TreatmentType'] in ['Surgery', 'Chemotherapy', 'Radiation']:
            if row['ChemotherapySessions'] + row['RadiationSessions'] > 15:
                return 'High'
            else:
                return 'Medium'
        elif row['TreatmentType'] == 'Combined':
            if row['ChemotherapySessions'] + row['RadiationSessions'] > 25:
                return 'High'
            else:
                return 'Medium'
        else:
            return 'Medium'
    
    df['TreatmentComplexity'] = df.apply(get_treatment_complexity, axis=1)
    
    # 3. Encoding categorical variables
    print("Encoding categorical variables...")
    
    # One-hot encoding for binary variables
    binary_vars = ['Gender', 'Metastasis']
    for var in binary_vars:
        dummies = pd.get_dummies(df[var], prefix=var, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    
    # Label encoding for ordinal variables
    ordinal_vars = {
        'CancerStage': {'I': 1, 'II': 2, 'III': 3, 'IV': 4},
        'SmokingStatus': {'Never': 0, 'Former': 1, 'Current': 2},
        'AlcoholUse': {'None': 0, 'Moderate': 1, 'Heavy': 2},
        'TreatmentComplexity': {'Low': 0, 'Medium': 1, 'High': 2}
    }
    
    for var, mapping in ordinal_vars.items():
        df[f"{var}_encoded"] = df[var].map(mapping)
    
    # For high-cardinality categorical variables, use a simpler encoding
    # TumorType encoding
    tumor_type_map = {tumor: idx for idx, tumor in enumerate(df['TumorType'].unique())}
    df['TumorType_encoded'] = df['TumorType'].map(tumor_type_map)
    
    # Treatment type encoding
    treatment_type_map = {treatment: idx for idx, treatment in enumerate(df['TreatmentType'].unique())}
    df['TreatmentType_encoded'] = df['TreatmentType'].map(treatment_type_map)
    
    # Target encoding for SurvivalStatus
    df['SurvivalStatus_Survived'] = (df['SurvivalStatus'] == 'Survived').astype(int)
    
    # 4. Feature scaling
    print("Scaling numerical features...")
    
    # Select numerical features for scaling
    numerical_features = ['Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 'FollowUpMonths', 'WaitingTimeDays']
    
    # Create a StandardScaler
    scaler = StandardScaler()
    
    # Fit and transform
    df[numerical_features] = pd.DataFrame(
        scaler.fit_transform(df[numerical_features]),
        columns=numerical_features,
        index=df.index
    )
    
    # Save the scaler for later use
    os.makedirs('../dashboard/models', exist_ok=True)
    joblib.dump(scaler, '../dashboard/models/scaler.pkl')
    
    print("Data preprocessing completed.")
    
    # Create a dataset for clustering
    clustering_features = [
        'Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions',
        'CancerStage_encoded', 'SmokingStatus_encoded', 'AlcoholUse_encoded',
        'Gender_Male', 'Metastasis_Yes', 'TumorType_encoded'
    ]
    
    # Create a dataset for classification
    classification_features = clustering_features + [
        'TreatmentType_encoded', 'WaitingTimeDays', 'TreatmentComplexity_encoded'
    ]
    
    # Save preprocessed data
    output_path = '../data/preprocessed_cancer_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    preprocess_data('../data/china_cancer_patients_synthetic.csv')