import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def build_logistic_regression_model(file_path):
    """
    Build and evaluate a logistic regression model for survival prediction.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the clustered data
    
    Returns:
    --------
    tuple
        (trained model, evaluation metrics, feature importance)
    """
    print(f"Loading clustered data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Select features for classification
    classification_features = [
        'Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions',
        'CancerStage_encoded', 'SmokingStatus_encoded', 'AlcoholUse_encoded',
        'Gender_Male', 'Metastasis_Yes', 'TumorType_encoded',
        'TreatmentType_encoded', 'WaitingTimeDays', 'TreatmentComplexity_encoded'
    ]
    
    # Define features
    X = df[classification_features]
    
    # Create proper target variable from SurvivalStatus
    print("\nCreating binary target variable from SurvivalStatus...")
    df['SurvivalStatus_Survived'] = (df['SurvivalStatus'] == 'Alive').astype(int)
    y = df['SurvivalStatus_Survived']
    
    # Check target variable distribution
    print("\nTarget variable distribution:")
    print(y.value_counts())
    
    # Check for missing values in features
    missing_values = X.isnull().sum()
    print("\nMissing values in features:")
    print(missing_values)
    
    # Handle missing values using SimpleImputer
    print("Imputing missing values...")
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Hyperparameter tuning with GridSearchCV
    print("Performing hyperparameter tuning...")
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Train the model with the best parameters
    print("Training the final model with the best parameters...")
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    
    # Save the trained model
    os.makedirs('../dashboard/models', exist_ok=True)
    joblib.dump(best_model, '../dashboard/models/logistic_regression_model.pkl')
    
    # Predictions on the test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    print("\nModel Evaluation:")
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
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("Creating confusion matrix visualization...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0.5, 1.5], ['Deceased', 'Survived'])
    plt.yticks([0.5, 1.5], ['Deceased', 'Survived'])
    plt.savefig('../dashboard/assets/confusion_matrix.png')
    
    # Save confusion matrix data for dashboard
    pd.DataFrame({
        'true_positive': [cm[1, 1]],
        'false_negative': [cm[1, 0]],
        'false_positive': [cm[0, 1]],
        'true_negative': [cm[0, 0]]
    }).to_csv('../dashboard/assets/confusion_matrix_data.csv', index=False)
    
    # Save evaluation metrics for dashboard
    pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC'],
        'Value': [accuracy, precision, recall, f1, roc_auc]
    }).to_csv('../dashboard/assets/model_metrics.csv', index=False)
    
    # ROC curve
    print("Creating ROC curve visualization...")
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
    plt.savefig('../dashboard/assets/roc_curve.png')
    
    # Save ROC curve data for dashboard
    pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr
    }).to_csv('../dashboard/assets/roc_curve_data.csv', index=False)
    
    # Feature importance analysis
    print("Analyzing feature importance...")
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': best_model.coef_[0],
        'Odds_Ratio': np.exp(best_model.coef_[0])
    })
    
    feature_importance['Absolute_Coefficient'] = np.abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Absolute_Coefficient', ascending=False)
    
    # Map feature names to more readable format
    feature_name_mapping = {
        'CancerStage_encoded': 'Cancer Stage',
        'Metastasis_Yes': 'Metastasis',
        'TreatmentType_encoded': 'Treatment Type',
        'TumorType_encoded': 'Tumor Type',
        'Age': 'Age',
        'TumorSize': 'Tumor Size',
        'SmokingStatus_encoded': 'Smoking Status',
        'AlcoholUse_encoded': 'Alcohol Use',
        'Gender_Male': 'Gender (Male)',
        'ChemotherapySessions': 'Chemotherapy Sessions',
        'RadiationSessions': 'Radiation Sessions',
        'WaitingTimeDays': 'Waiting Time (Days)',
        'TreatmentComplexity_encoded': 'Treatment Complexity'
    }
    
    feature_importance['Feature_Name'] = feature_importance['Feature'].map(
        lambda x: feature_name_mapping.get(x, x)
    )
    
    # Save feature importance for dashboard
    feature_importance[['Feature_Name', 'Coefficient', 'Odds_Ratio', 'Absolute_Coefficient']].to_csv(
        '../dashboard/assets/feature_importance.csv', index=False
    )
    
    # Visualize feature importance
    print("Creating feature importance visualization...")
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(10)
    colors = ['red' if c < 0 else 'green' for c in top_features['Coefficient']]
    
    plt.barh(top_features['Feature_Name'], top_features['Absolute_Coefficient'], color=colors)
    plt.title('Top 10 Features by Importance')
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Feature')
    plt.savefig('../dashboard/assets/feature_importance.png')
    
    # Create coefficients and intercept data file for prediction in dashboard
    coef_data = {
        'intercept': best_model.intercept_[0],
        'coefficients': {feature: coef for feature, coef in zip(X_train.columns, best_model.coef_[0])}
    }
    
    import json
    with open('../dashboard/assets/model_coefficients.json', 'w') as f:
        json.dump(coef_data, f)
    
    print("Logistic regression model analysis completed.")
    
    # Return the model and evaluation metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    return best_model, metrics, feature_importance

if __name__ == "__main__":
    build_logistic_regression_model('../data/clustered_cancer_data.csv')