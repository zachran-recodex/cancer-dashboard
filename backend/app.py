import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from config import config

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.config.from_object(config['development'])

# Enable CORS
CORS(app)

# Load dataset
@app.route('/api/data/summary', methods=['GET'])
def get_data_summary():
    try:
        # In a real app, you would load the actual data file
        # For now, we'll return mock data
        data = {
            'total_patients': 10000,
            'gender_distribution': {'Male': 5486, 'Female': 4514},
            'cancer_stages': {'I': 1874, 'II': 2786, 'III': 3240, 'IV': 2100},
            'survival_rate': 68.42
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Age distribution
@app.route('/api/data/age-distribution', methods=['GET'])
def get_age_distribution():
    try:
        # Mock data for age distribution
        data = [
            {'ageGroup': '<30', 'count': 153},
            {'ageGroup': '30-39', 'count': 639},
            {'ageGroup': '40-49', 'count': 1686},
            {'ageGroup': '50-59', 'count': 2721},
            {'ageGroup': '60-69', 'count': 2638},
            {'ageGroup': '70-79', 'count': 1511},
            {'ageGroup': '80+', 'count': 652}
        ]
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Cancer types by gender
@app.route('/api/data/cancer-by-gender', methods=['GET'])
def get_cancer_by_gender():
    try:
        # Mock data for cancer types by gender
        data = [
            {'type': 'Lung', 'male': 1264, 'female': 1033},
            {'type': 'Colorectal', 'male': 1050, 'female': 894},
            {'type': 'Breast', 'male': 813, 'female': 727},
            {'type': 'Liver', 'male': 689, 'female': 600},
            {'type': 'Stomach', 'male': 492, 'female': 432},
            {'type': 'Others', 'male': 1107, 'female': 899}
        ]
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Cluster data
@app.route('/api/data/clusters', methods=['GET'])
def get_clusters():
    try:
        # Mock cluster data
        data = {
            'clusterDistribution': [
                {'name': 'Young Survivors', 'value': 3246},
                {'name': 'Mid-stage Patients', 'value': 4128},
                {'name': 'Advanced Cases', 'value': 2626}
            ],
            'clusterProfile': [
                {'axis': 'Age', 'cluster1': 45.6, 'cluster2': 59.5, 'cluster3': 74.5},
                {'axis': 'Tumor Size', 'cluster1': 2.2, 'cluster2': 4.1, 'cluster3': 7.7},
                {'axis': 'Cancer Stage', 'cluster1': 1.5, 'cluster2': 2.5, 'cluster3': 3.5},
                {'axis': 'Metastasis', 'cluster1': 0.1, 'cluster2': 0.3, 'cluster3': 0.8},
                {'axis': 'Smoking Status', 'cluster1': 0.6, 'cluster2': 1.4, 'cluster3': 1.8},
                {'axis': 'Alcohol Use', 'cluster1': 0.5, 'cluster2': 1.1, 'cluster3': 1.8},
                {'axis': 'Survival Rate', 'cluster1': 0.71, 'cluster2': 0.68, 'cluster3': 0.24}
            ]
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Model performance data
@app.route('/api/data/model-performance', methods=['GET'])
def get_model_performance():
    try:
        # Mock model performance data
        data = {
            'accuracy': 87.8,
            'precision': 90.2,
            'recall': 92.3,
            'f1Score': 91.2,
            'auc': 89.0,
            'featureImportance': [
                {'feature': 'Cancer Stage IV', 'importance': 1.87},
                {'feature': 'Metastasis', 'importance': 1.64},
                {'feature': 'Cancer Stage III', 'importance': 1.22},
                {'feature': 'Combined Treatment', 'importance': 1.25},
                {'feature': 'Current Smoking', 'importance': 0.85},
                {'feature': 'Surgery', 'importance': 0.87},
                {'feature': 'Lung Cancer', 'importance': 0.72},
                {'feature': 'Age', 'importance': 0.59},
                {'feature': 'Liver Cancer', 'importance': 0.54},
                {'feature': 'Tumor Size', 'importance': 0.49}
            ]
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Predict survival probability
@app.route('/api/predict', methods=['POST'])
def predict_survival():
    try:
        data = request.json
        
        # In a real app, you would use a trained model for prediction
        # For now, we'll return mock prediction
        
        # Simple mock logic based on cancer stage
        stage = data.get('cancerStage', 2)
        if stage >= 3:
            probability = 0.35
            prediction = "Deceased"
        else:
            probability = 0.72
            prediction = "Survived"
            
        result = {
            'prediction': prediction,
            'probability': probability,
            'factors': [
                {'factor': 'Early stage cancer', 'impact': 'positive' if stage < 3 else 'negative'},
                {'factor': 'Combined treatment approach', 'impact': 'positive'},
                {'factor': 'No metastasis detected', 'impact': 'positive' if not data.get('metastasis', False) else 'negative'},
                {'factor': 'Moderate alcohol consumption', 'impact': 'negative'},
                {'factor': 'Current smoking status', 'impact': 'negative'}
            ]
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Serve React app in production
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
