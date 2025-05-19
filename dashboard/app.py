import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import os
import json
from plotly.subplots import make_subplots

# Initialize the Dash app
app = dash.Dash(__name__, assets_folder='assets')
server = app.server
app.title = 'Cancer Patient Analysis Dashboard'

# Load the data
try:
    # Check if all required files exist
    required_files = [
        '../data/china_cancer_patients_synthetic.csv',
        '../data/clustered_cancer_data.csv',
        'assets/cluster_names.csv',
        'assets/feature_importance.csv',
        'assets/model_metrics.csv',
        'assets/confusion_matrix_data.csv',
        'assets/roc_curve_data.csv',
        'assets/model_coefficients.json',
        'models/kmeans_model.pkl',
        'models/logistic_regression_model.pkl',
        'models/scaler.pkl'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Warning: The following files are missing: {missing_files}")
        print("The dashboard may not function correctly. Please run the data preparation and modeling scripts first.")
    
    # Load the original dataset
    df_original = pd.read_csv('../data/china_cancer_patients_synthetic.csv')
    
    # Load the processed dataset with cluster labels
    df_clustered = pd.read_csv('../data/clustered_cancer_data.csv')
    
    # Load cluster names
    df_cluster_names = pd.read_csv('assets/cluster_names.csv')
    cluster_names = {row['Cluster']: row['Name'] for _, row in df_cluster_names.iterrows()}
    
    # Load feature importance
    df_feature_importance = pd.read_csv('assets/feature_importance.csv')
    
    # Load model metrics
    df_model_metrics = pd.read_csv('assets/model_metrics.csv')
    
    # Load confusion matrix data
    df_confusion_matrix = pd.read_csv('assets/confusion_matrix_data.csv')
    
    # Load ROC curve data
    df_roc_curve = pd.read_csv('assets/roc_curve_data.csv')
    
    # Load model coefficients for prediction
    with open('assets/model_coefficients.json', 'r') as f:
        model_coefficients = json.load(f)
    
    # Load models
    kmeans_model = joblib.load('models/kmeans_model.pkl')
    logreg_model = joblib.load('models/logistic_regression_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    print("All data and models loaded successfully.")
    
except Exception as e:
    print(f"Error loading data or models: {e}")
    # Create empty dataframes if data loading fails
    df_original = pd.DataFrame()
    df_clustered = pd.DataFrame()
    cluster_names = {0: "Cluster 0", 1: "Cluster 1", 2: "Cluster 2"}
    df_feature_importance = pd.DataFrame()
    df_model_metrics = pd.DataFrame()
    df_confusion_matrix = pd.DataFrame()
    df_roc_curve = pd.DataFrame()
    model_coefficients = {"intercept": 0, "coefficients": {}}

# Define the layout of the app
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Cancer Patient Analysis Dashboard", className="header-title"),
        html.P("Analysis of 10,000 Cancer Patients in China using Data Mining Techniques", className="header-description"),
    ], className="header"),
    
    # Navigation Tabs
    dcc.Tabs(id="tabs", value="tab-overview", children=[
        # Overview Tab
        dcc.Tab(label="Overview", value="tab-overview", children=[
            html.Div([
                html.H2("Dataset Overview", className="section-title"),
                
                # Dataset Summary
                html.Div([
                    html.Div([
                        html.H3("Dataset Statistics"),
                        html.Div(id="dataset-stats", className="stats-container")
                    ], className="card"),
                    
                    html.Div([
                        html.H3("Data Distribution"),
                        dcc.Dropdown(
                            id="overview-feature-dropdown",
                            options=[
                                {"label": "Age", "value": "Age"},
                                {"label": "Gender", "value": "Gender"},
                                {"label": "Tumor Type", "value": "TumorType"},
                                {"label": "Cancer Stage", "value": "CancerStage"},
                                {"label": "Treatment Type", "value": "TreatmentType"},
                                {"label": "Survival Status", "value": "SurvivalStatus"},
                            ],
                            value="Age",
                            clearable=False
                        ),
                        dcc.Graph(id="overview-distribution-plot")
                    ], className="card")
                ], className="grid-container"),
                
                # Key Relationships
                html.Div([
                    html.H3("Key Relationships", className="section-title"),
                    html.Div([
                        html.Div([
                            html.H4("Cancer Types by Gender"),
                            dcc.Graph(id="cancer-by-gender-plot")
                        ], className="card"),
                        
                        html.Div([
                            html.H4("Survival Rate by Cancer Stage"),
                            dcc.Graph(id="survival-by-stage-plot")
                        ], className="card"),
                    ], className="grid-container"),
                ]),
            ], className="tab-content")
        ]),
        
        # Clustering Results Tab
        dcc.Tab(label="Clustering Results", value="tab-clustering", children=[
            html.Div([
                html.H2("K-means Clustering Analysis", className="section-title"),
                
                html.Div([
                    html.Div([
                        html.H3("Cluster Distribution"),
                        dcc.Graph(id="cluster-distribution-plot")
                    ], className="card"),
                    
                    html.Div([
                        html.H3("Cluster Visualization"),
                        dcc.Graph(id="cluster-visualization-plot")
                    ], className="card"),
                ], className="grid-container"),
                
                html.Div([
                    html.H3("Cluster Profiles", className="section-title"),
                    html.Div([
                        html.Div([
                            html.H4("Select Cluster:"),
                            dcc.Dropdown(
                                id="cluster-dropdown",
                                options=[
                                    {"label": f"Cluster {i}: {name}", "value": i} 
                                    for i, name in cluster_names.items()
                                ],
                                value=0,
                                clearable=False
                            ),
                            html.Div(id="cluster-profile-card")
                        ], className="card"),
                        
                        html.Div([
                            html.H4("Comparative Analysis"),
                            dcc.Dropdown(
                                id="comparative-feature-dropdown",
                                options=[
                                    {"label": "Age", "value": "Age"},
                                    {"label": "Tumor Size", "value": "TumorSize"},
                                    {"label": "Chemo Sessions", "value": "ChemotherapySessions"},
                                    {"label": "Radiation Sessions", "value": "RadiationSessions"},
                                    {"label": "Follow-up Months", "value": "FollowUpMonths"},
                                ],
                                value="Age",
                                clearable=False
                            ),
                            dcc.Graph(id="comparative-cluster-plot")
                        ], className="card"),
                    ], className="grid-container"),
                ]),
            ], className="tab-content")
        ]),
        
        # Prediction Model Tab
        dcc.Tab(label="Survival Prediction", value="tab-prediction", children=[
            html.Div([
                html.H2("Logistic Regression Model for Survival Prediction", className="section-title"),
                
                html.Div([
                    html.Div([
                        html.H3("Model Performance"),
                        html.Div(id="model-metrics-display", className="metrics-container")
                    ], className="card"),
                    
                    html.Div([
                        html.H3("Feature Importance"),
                        dcc.Graph(id="feature-importance-plot")
                    ], className="card"),
                ], className="grid-container"),
                
                html.Div([
                    html.Div([
                        html.H3("Confusion Matrix"),
                        dcc.Graph(id="confusion-matrix-plot")
                    ], className="card"),
                    
                    html.Div([
                        html.H3("ROC Curve"),
                        dcc.Graph(id="roc-curve-plot")
                    ], className="card"),
                ], className="grid-container"),
                
                html.Div([
                    html.H3("Survival Probability Calculator", className="section-title"),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Age:"),
                                dcc.Input(id="input-age", type="number", value=60, min=18, max=95, className="input-field")
                            ], className="input-group"),
                            
                            html.Div([
                                html.Label("Gender:"),
                                dcc.Dropdown(
                                    id="input-gender",
                                    options=[
                                        {"label": "Male", "value": "Male"},
                                        {"label": "Female", "value": "Female"}
                                    ],
                                    value="Male",
                                    clearable=False,
                                    className="dropdown-field"
                                )
                            ], className="input-group"),
                            
                            html.Div([
                                html.Label("Tumor Type:"),
                                dcc.Dropdown(
                                    id="input-tumor-type",
                                    options=[
                                        {"label": "Lung", "value": "Lung"},
                                        {"label": "Colorectal", "value": "Colorectal"},
                                        {"label": "Breast", "value": "Breast"},
                                        {"label": "Liver", "value": "Liver"},
                                        {"label": "Stomach", "value": "Stomach"},
                                        {"label": "Other", "value": "Other"}
                                    ],
                                    value="Lung",
                                    clearable=False,
                                    className="dropdown-field"
                                )
                            ], className="input-group"),
                            
                            html.Div([
                                html.Label("Cancer Stage:"),
                                dcc.Dropdown(
                                    id="input-cancer-stage",
                                    options=[
                                        {"label": "Stage I", "value": "I"},
                                        {"label": "Stage II", "value": "II"},
                                        {"label": "Stage III", "value": "III"},
                                        {"label": "Stage IV", "value": "IV"}
                                    ],
                                    value="II",
                                    clearable=False,
                                    className="dropdown-field"
                                )
                            ], className="input-group"),
                        ], className="input-column"),
                        
                        html.Div([
                            html.Div([
                                html.Label("Tumor Size (cm):"),
                                dcc.Input(id="input-tumor-size", type="number", value=4.0, min=0.3, max=18.5, step=0.1, className="input-field")
                            ], className="input-group"),
                            
                            html.Div([
                                html.Label("Metastasis:"),
                                dcc.Dropdown(
                                    id="input-metastasis",
                                    options=[
                                        {"label": "Yes", "value": "Yes"},
                                        {"label": "No", "value": "No"}
                                    ],
                                    value="No",
                                    clearable=False,
                                    className="dropdown-field"
                                )
                            ], className="input-group"),
                            
                            html.Div([
                                html.Label("Treatment Type:"),
                                dcc.Dropdown(
                                    id="input-treatment-type",
                                    options=[
                                        {"label": "Surgery", "value": "Surgery"},
                                        {"label": "Chemotherapy", "value": "Chemotherapy"},
                                        {"label": "Radiation", "value": "Radiation"},
                                        {"label": "Combined", "value": "Combined"},
                                        {"label": "Palliative", "value": "Palliative"}
                                    ],
                                    value="Combined",
                                    clearable=False,
                                    className="dropdown-field"
                                )
                            ], className="input-group"),
                            
                            html.Div([
                                html.Label("Chemo Sessions:"),
                                dcc.Input(id="input-chemo-sessions", type="number", value=6, min=0, max=12, className="input-field")
                            ], className="input-group"),
                        ], className="input-column"),
                        
                        html.Div([
                            html.Div([
                                html.Label("Radiation Sessions:"),
                                dcc.Input(id="input-radiation-sessions", type="number", value=10, min=0, max=35, className="input-field")
                            ], className="input-group"),
                            
                            html.Div([
                                html.Label("Smoking Status:"),
                                dcc.Dropdown(
                                    id="input-smoking-status",
                                    options=[
                                        {"label": "Never", "value": "Never"},
                                        {"label": "Former", "value": "Former"},
                                        {"label": "Current", "value": "Current"}
                                    ],
                                    value="Former",
                                    clearable=False,
                                    className="dropdown-field"
                                )
                            ], className="input-group"),
                            
                            html.Div([
                                html.Label("Alcohol Use:"),
                                dcc.Dropdown(
                                    id="input-alcohol-use",
                                    options=[
                                        {"label": "None", "value": "None"},
                                        {"label": "Moderate", "value": "Moderate"},
                                        {"label": "Heavy", "value": "Heavy"}
                                    ],
                                    value="Moderate",
                                    clearable=False,
                                    className="dropdown-field"
                                )
                            ], className="input-group"),
                            
                            html.Button("Predict Survival", id="predict-button", className="predict-button")
                        ], className="input-column"),
                    ], className="calculator-container"),
                    
                    # Prediction results
                    html.Div(id="prediction-results", className="prediction-results")
                ], className="card prediction-card"),
            ], className="tab-content")
        ]),
    ]),
    
    # Footer
    html.Footer([
        html.P("Data Mining Project - Analysis of Cancer Patients in China"),
        html.P("Telkom University - Faculty of Informatics - 2025"),
    ], className="footer")
], className="app-container")

# Callback for dataset statistics
@app.callback(
    Output("dataset-stats", "children"),
    Input("tabs", "value")
)
def update_dataset_stats(tab):
    if df_original.empty:
        return html.Div("Data not available", className="error-message")
    
    # Get survival rate more safely by checking actual values first
    survival_rate = 0
    if 'SurvivalStatus' in df_original.columns:
        status_counts = df_original['SurvivalStatus'].value_counts()
        
        # Try to find positive survival value by checking common terms
        survival_terms = ['Survived', 'survived', 'Alive', 'alive', 'Yes', 'yes', '1', 1, True]
        
        for term in survival_terms:
            if term in status_counts:
                survival_rate = status_counts[term] / len(df_original) * 100
                break
        
        # If we couldn't find a match, just use the first value as a fallback
        if survival_rate == 0 and not status_counts.empty:
            survival_rate = status_counts.iloc[0] / len(df_original) * 100
    
    return html.Div([
        html.Div([
            html.Div(f"{len(df_original)}", className="stat-value"),
            html.Div("Patients", className="stat-label")
        ], className="stat-item"),
        
        html.Div([
            html.Div(f"{df_original['Gender'].value_counts().iloc[0] / len(df_original) * 100:.1f}%", className="stat-value"),
            html.Div(f"{df_original['Gender'].value_counts().index[0]}", className="stat-label")
        ], className="stat-item"),
        
        html.Div([
            html.Div(f"{df_original['Age'].mean():.1f}", className="stat-value"),
            html.Div("Avg. Age", className="stat-label")
        ], className="stat-item"),
        
        html.Div([
            html.Div(f"{survival_rate:.1f}%", className="stat-value"),
            html.Div("Survival Rate", className="stat-label")
        ], className="stat-item"),
    ])

# Callback for overview distribution plot
@app.callback(
    Output("overview-distribution-plot", "figure"),
    Input("overview-feature-dropdown", "value")
)
def update_distribution_plot(feature):
    if df_original.empty:
        return go.Figure()
    
    if feature in ["Age", "TumorSize", "ChemotherapySessions", "RadiationSessions", "FollowUpMonths"]:
        # Histogram for numerical features
        fig = px.histogram(
            df_original, 
            x=feature,
            title=f"Distribution of {feature}",
            color_discrete_sequence=["#3366CC"],
            opacity=0.8
        )
        fig.update_layout(
            xaxis_title=feature,
            yaxis_title="Count",
            margin=dict(l=40, r=40, t=40, b=40),
            template="plotly_white"
        )
    else:
        # Bar chart for categorical features
        # Buat DataFrame terlebih dahulu dan beri nama kolom yang eksplisit
        value_counts_df = df_original[feature].value_counts().reset_index()
        value_counts_df.columns = [feature, 'count']  # Nama kolom yang jelas
        
        fig = px.bar(
            value_counts_df,
            x=feature,
            y='count',
            title=f"Distribution of {feature}",
            color_discrete_sequence=["#3366CC"]
        )
        fig.update_layout(
            xaxis_title=feature,
            yaxis_title="Count",
            margin=dict(l=40, r=40, t=40, b=40),
            template="plotly_white"
        )
    
    return fig

# Callback for cancer by gender plot
@app.callback(
    Output("cancer-by-gender-plot", "figure"),
    Input("tabs", "value")
)
def update_cancer_by_gender_plot(tab):
    if df_original.empty:
        return go.Figure()
    
    # Group by TumorType and Gender
    tumor_gender_counts = df_original.groupby(['TumorType', 'Gender']).size().reset_index(name='Count')
    
    # Create grouped bar chart
    fig = px.bar(
        tumor_gender_counts,
        x="TumorType",
        y="Count",
        color="Gender",
        barmode="group",
        color_discrete_map={"Male": "#3366CC", "Female": "#FF6B6B"},
        labels={"TumorType": "Tumor Type", "Count": "Number of Patients"}
    )
    
    fig.update_layout(
        xaxis_title="Tumor Type",
        yaxis_title="Number of Patients",
        legend_title="Gender",
        margin=dict(l=40, r=40, t=20, b=40),
        template="plotly_white"
    )
    
    return fig

# Callback for survival by stage plot
@app.callback(
    Output("survival-by-stage-plot", "figure"),
    Input("tabs", "value")
)
def update_survival_by_stage_plot(tab):
    if df_original.empty:
        return go.Figure()
    
    # Calculate survival rates by cancer stage
    survival_by_stage = df_original.groupby('CancerStage')['SurvivalStatus'].apply(
        lambda x: (x == 'Survived').mean() * 100
    ).reset_index(name='SurvivalRate')
    
    # Sort by stage
    stage_order = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
    survival_by_stage['StageOrder'] = survival_by_stage['CancerStage'].map(stage_order)
    survival_by_stage = survival_by_stage.sort_values('StageOrder')
    
    # Create bar chart
    fig = px.bar(
        survival_by_stage,
        x="CancerStage",
        y="SurvivalRate",
        color="SurvivalRate",
        text=survival_by_stage["SurvivalRate"].round(1).astype(str) + "%",
        color_continuous_scale=["#FF6B6B", "#3366CC"],
        labels={"CancerStage": "Cancer Stage", "SurvivalRate": "Survival Rate (%)"}
    )
    
    fig.update_layout(
        xaxis_title="Cancer Stage",
        yaxis_title="Survival Rate (%)",
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=20, b=40),
        template="plotly_white"
    )
    
    fig.update_traces(textposition='outside')
    
    return fig

# Callback for cluster distribution plot
@app.callback(
    Output("cluster-distribution-plot", "figure"),
    Input("tabs", "value")
)
def update_cluster_distribution_plot(tab):
    if df_clustered.empty:
        return go.Figure()
    
    # Calculate the count and percentage of patients in each cluster
    cluster_counts = df_clustered['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    cluster_counts['Percentage'] = cluster_counts['Count'] / cluster_counts['Count'].sum() * 100
    
    # Add cluster names
    cluster_counts['ClusterName'] = cluster_counts['Cluster'].map(cluster_names)
    cluster_counts['Label'] = cluster_counts['ClusterName'] + " (" + cluster_counts['Percentage'].round(1).astype(str) + "%)"
    
    # Create pie chart
    fig = px.pie(
        cluster_counts,
        values='Count',
        names='Label',
        color='Cluster',
        color_discrete_sequence=["#3366CC", "#FF6B6B", "#33CC99"],
        hole=0.4
    )
    
    fig.update_layout(
        legend_title="Clusters",
        margin=dict(l=20, r=20, t=20, b=20),
        template="plotly_white"
    )
    
    return fig

# Callback for cluster visualization plot
@app.callback(
    Output("cluster-visualization-plot", "figure"),
    Input("tabs", "value")
)
def update_cluster_visualization_plot(tab):
    if df_clustered.empty:
        return go.Figure()
    
    try:
        # Load PCA results if available
        pca_results = pd.read_csv('assets/pca_results.csv')
        
        # Add cluster names
        pca_results['ClusterName'] = pca_results['Cluster'].map(cluster_names)
        
        # Create scatter plot
        fig = px.scatter(
            pca_results,
            x="PCA1",
            y="PCA2",
            color="ClusterName",
            color_discrete_sequence=["#3366CC", "#FF6B6B", "#33CC99"],
            labels={"ClusterName": "Cluster"},
            opacity=0.7
        )
        
        fig.update_layout(
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            legend_title="Clusters",
            margin=dict(l=40, r=40, t=20, b=40),
            template="plotly_white"
        )
        
    except Exception as e:
        print(f"Error creating cluster visualization: {e}")
        # If PCA results are not available, create a basic scatter plot
        fig = px.scatter(
            df_clustered,
            x="Age",
            y="TumorSize",
            color="Cluster",
            color_discrete_sequence=["#3366CC", "#FF6B6B", "#33CC99"],
            labels={"Cluster": "Cluster"}
        )
        
        fig.update_layout(
            xaxis_title="Age (standardized)",
            yaxis_title="Tumor Size (standardized)",
            legend_title="Clusters",
            margin=dict(l=40, r=40, t=20, b=40),
            template="plotly_white"
        )
    
    return fig

# Callback for cluster profile card
@app.callback(
    Output("cluster-profile-card", "children"),
    Input("cluster-dropdown", "value")
)
def update_cluster_profile(cluster_id):
    if df_clustered.empty:
        return html.Div("Data not available", className="error-message")
    
    # Filter data for the selected cluster
    cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
    
    if cluster_data.empty:
        return html.Div("No data for this cluster", className="error-message")
    
    # Calculate cluster statistics
    avg_age = df_original['Age'].loc[cluster_data.index].mean()
    avg_tumor_size = df_original['TumorSize'].loc[cluster_data.index].mean()
    
    # Find the positive survival value
    survival_terms = ['Survived', 'survived', 'Alive', 'alive', 'Yes', 'yes', '1', 1, True]
    survival_value = None
    
    for term in survival_terms:
        if term in df_original['SurvivalStatus'].unique():
            survival_value = term
            break
    
    # If we couldn't find a match, just use the first value
    if survival_value is None and len(df_original['SurvivalStatus'].unique()) > 0:
        survival_value = df_original['SurvivalStatus'].unique()[0]
    
    # Calculate survival rate
    if survival_value is not None:
        survival_rate = (df_original['SurvivalStatus'].loc[cluster_data.index] == survival_value).mean() * 100
    else:
        survival_rate = 0
    
    # Get dominant tumor types
    tumor_counts = df_original['TumorType'].loc[cluster_data.index].value_counts()
    dominant_tumors = tumor_counts.head(2).index.tolist()
    dominant_tumors_str = f"{dominant_tumors[0]} ({tumor_counts[dominant_tumors[0]] / len(cluster_data) * 100:.1f}%)"
    if len(dominant_tumors) > 1:
        dominant_tumors_str += f", {dominant_tumors[1]} ({tumor_counts[dominant_tumors[1]] / len(cluster_data) * 100:.1f}%)"
    
    # Get dominant cancer stages
    stage_counts = df_original['CancerStage'].loc[cluster_data.index].value_counts()
    dominant_stages = stage_counts.head(2).index.tolist()
    dominant_stages_str = f"{dominant_stages[0]} ({stage_counts[dominant_stages[0]] / len(cluster_data) * 100:.1f}%)"
    if len(dominant_stages) > 1:
        dominant_stages_str += f", {dominant_stages[1]} ({stage_counts[dominant_stages[1]] / len(cluster_data) * 100:.1f}%)"
    
    # Get dominant treatment types
    treatment_counts = df_original['TreatmentType'].loc[cluster_data.index].value_counts()
    dominant_treatments = treatment_counts.head(2).index.tolist()
    dominant_treatments_str = f"{dominant_treatments[0]} ({treatment_counts[dominant_treatments[0]] / len(cluster_data) * 100:.1f}%)"
    if len(dominant_treatments) > 1:
        dominant_treatments_str += f", {dominant_treatments[1]} ({treatment_counts[dominant_treatments[1]] / len(cluster_data) * 100:.1f}%)"
    
    # Metastasis rate - handle different possible values
    metastasis_terms = ['Yes', 'yes', 'True', 'true', '1', 1, True]
    metastasis_value = None
    
    for term in metastasis_terms:
        if term in df_original['Metastasis'].unique():
            metastasis_value = term
            break
    
    # If we couldn't find a match, just use the first value different from 'No'
    if metastasis_value is None:
        for val in df_original['Metastasis'].unique():
            if val not in ['No', 'no', 'False', 'false', '0', 0, False]:
                metastasis_value = val
                break
    
    # Calculate metastasis rate
    if metastasis_value is not None:
        metastasis_rate = (df_original['Metastasis'].loc[cluster_data.index] == metastasis_value).mean() * 100
    else:
        metastasis_rate = 0

    return html.Div([
        html.H4(f"Cluster {cluster_id}: {cluster_names[cluster_id]}", className="cluster-name"),
        
        html.Div([
            html.Div([
                html.Span("Patient Count:", className="profile-label"),
                html.Span(f"{len(cluster_data):,} ({len(cluster_data) / len(df_clustered) * 100:.1f}%)", className="profile-value")
            ], className="profile-item"),
            
            html.Div([
                html.Span("Average Age:", className="profile-label"),
                html.Span(f"{avg_age:.1f} years", className="profile-value")
            ], className="profile-item"),
            
            html.Div([
                html.Span("Average Tumor Size:", className="profile-label"),
                html.Span(f"{avg_tumor_size:.1f} cm", className="profile-value")
            ], className="profile-item"),
            
            html.Div([
                html.Span("Survival Rate:", className="profile-label"),
                html.Span(f"{survival_rate:.1f}%", className="profile-value")
            ], className="profile-item"),
            
            html.Div([
                html.Span("Dominant Tumor Types:", className="profile-label"),
                html.Span(dominant_tumors_str, className="profile-value")
            ], className="profile-item"),
            
            html.Div([
                html.Span("Dominant Cancer Stages:", className="profile-label"),
                html.Span(dominant_stages_str, className="profile-value")
            ], className="profile-item"),
            
            html.Div([
                html.Span("Dominant Treatment Types:", className="profile-label"),
                html.Span(dominant_treatments_str, className="profile-value")
            ], className="profile-item"),
            
            html.Div([
                html.Span("Metastasis Rate:", className="profile-label"),
                html.Span(f"{metastasis_rate:.1f}%", className="profile-value")
            ], className="profile-item"),
        ], className="profile-container"),
        
        html.Div([
            html.P("Cluster Characteristics:", className="characteristics-title"),
            html.P(get_cluster_description(cluster_id), className="characteristics-text")
        ], className="characteristics-container")
    ])

# Function to get cluster descriptions
def get_cluster_description(cluster_id):
    descriptions = {
        0: "This cluster primarily consists of younger patients with early-stage cancer (mainly Stage I and II). They typically have smaller tumor sizes, low metastasis rates, and a high survival rate. The most common treatment is surgery, followed by combined therapy approaches. Patients in this group are less likely to smoke or consume alcohol heavily.",
        
        1: "This cluster represents middle-aged patients with intermediate-stage cancer (mainly Stage II and III). They have moderate tumor sizes and metastasis rates. The most common treatment is a combined approach. This group shows a moderate survival rate and more variable health behaviors regarding smoking and alcohol consumption.",
        
        2: "This cluster contains predominantly elderly patients with advanced-stage cancer (mainly Stage III and IV). They have larger tumor sizes and high metastasis rates. Palliative care and radiation therapy are more common in this group. These patients show the lowest survival rates and higher rates of smoking and heavy alcohol consumption."
    }
    
    return descriptions.get(cluster_id, "No description available for this cluster.")

# Callback for comparative cluster plot
@app.callback(
    Output("comparative-cluster-plot", "figure"),
    Input("comparative-feature-dropdown", "value"),
    Input("tabs", "value")
)
def update_comparative_cluster_plot(feature, tab):
    if df_clustered.empty or df_original.empty:
        return go.Figure()
    
    # Combine cluster information with original data
    df_combined = df_original.copy()
    df_combined['Cluster'] = df_clustered['Cluster']
    df_combined['ClusterName'] = df_combined['Cluster'].map(cluster_names)
    
    # Create box plot
    fig = px.box(
        df_combined,
        x="ClusterName",
        y=feature,
        color="ClusterName",
        color_discrete_sequence=["#3366CC", "#FF6B6B", "#33CC99"],
        labels={"ClusterName": "Cluster", feature: feature},
        notched=True
    )
    
    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title=feature,
        legend_title="Clusters",
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=40),
        template="plotly_white"
    )
    
    return fig

# Callback for model metrics display
@app.callback(
    Output("model-metrics-display", "children"),
    Input("tabs", "value")
)
def update_model_metrics_display(tab):
    if df_model_metrics.empty:
        return html.Div("Model metrics not available", className="error-message")
    
    # Create a metrics display
    metrics = {}
    for _, row in df_model_metrics.iterrows():
        metrics[row['Metric']] = row['Value']
    
    return html.Div([
        html.Div([
            html.Div(f"{metrics.get('Accuracy', 0) * 100:.1f}%", className="metric-value"),
            html.Div("Accuracy", className="metric-label")
        ], className="metric-item"),
        
        html.Div([
            html.Div(f"{metrics.get('Precision', 0) * 100:.1f}%", className="metric-value"),
            html.Div("Precision", className="metric-label")
        ], className="metric-item"),
        
        html.Div([
            html.Div(f"{metrics.get('Recall', 0) * 100:.1f}%", className="metric-value"),
            html.Div("Recall", className="metric-label")
        ], className="metric-item"),
        
        html.Div([
            html.Div(f"{metrics.get('F1-score', 0) * 100:.1f}%", className="metric-value"),
            html.Div("F1-Score", className="metric-label")
        ], className="metric-item"),
        
        html.Div([
            html.Div(f"{metrics.get('AUC-ROC', 0) * 100:.1f}%", className="metric-value"),
            html.Div("AUC-ROC", className="metric-label")
        ], className="metric-item"),
    ], className="metrics-grid")

# Callback for feature importance plot
@app.callback(
    Output("feature-importance-plot", "figure"),
    Input("tabs", "value")
)
def update_feature_importance_plot(tab):
    if df_feature_importance.empty:
        return go.Figure()
    
    # Get top features
    top_features = df_feature_importance.head(10).copy()
    top_features = top_features.sort_values('Absolute_Coefficient')
    
    # Create bar chart
    colors = ['red' if c < 0 else 'green' for c in top_features['Coefficient']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=top_features['Feature_Name'],
        x=top_features['Absolute_Coefficient'],
        orientation='h',
        marker_color=colors,
        text=top_features['Coefficient'].round(3),
        textposition='auto',
        hovertemplate=
        '<b>%{y}</b><br>' +
        'Coefficient: %{text}<br>' +
        'Odds Ratio: %{customdata}<br>' +
        '<extra></extra>',
        customdata=np.exp(top_features['Coefficient']).round(3)
    ))
    
    fig.update_layout(
        title="Top 10 Features by Importance",
        xaxis_title="Absolute Coefficient Value",
        yaxis_title="Feature",
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white"
    )
    
    return fig

# Callback for confusion matrix plot
@app.callback(
    Output("confusion-matrix-plot", "figure"),
    Input("tabs", "value")
)
def update_confusion_matrix_plot(tab):
    if df_confusion_matrix.empty:
        return go.Figure()
    
    # Get confusion matrix values
    try:
        cm = [
            [df_confusion_matrix['true_negative'].iloc[0], df_confusion_matrix['false_positive'].iloc[0]],
            [df_confusion_matrix['false_negative'].iloc[0], df_confusion_matrix['true_positive'].iloc[0]]
        ]
        
        # Create annotated heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted: Deceased', 'Predicted: Survived'],
            y=['Actual: Deceased', 'Actual: Survived'],
            colorscale='Blues',
            showscale=False
        ))
        
        # Add text annotations
        annotations = []
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                annotations.append({
                    'x': j,
                    'y': i,
                    'text': str(cm[i][j]),
                    'font': {'color': 'white' if cm[i][j] > 200 else 'black'},
                    'showarrow': False
                })
        
        fig.update_layout(
            annotations=annotations,
            margin=dict(l=40, r=40, t=20, b=40),
            template="plotly_white"
        )
        
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        fig = go.Figure()
    
    return fig

# Callback for ROC curve plot
@app.callback(
    Output("roc-curve-plot", "figure"),
    Input("tabs", "value")
)
def update_roc_curve_plot(tab):
    if df_roc_curve.empty:
        return go.Figure()
    
    # Get AUC-ROC value
    auc_roc = 0
    if not df_model_metrics.empty:
        auc_row = df_model_metrics[df_model_metrics['Metric'] == 'AUC-ROC']
        if not auc_row.empty:
            auc_roc = auc_row['Value'].iloc[0]
    
    # Create ROC curve
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_roc_curve['fpr'],
        y=df_roc_curve['tpr'],
        mode='lines',
        name=f'ROC (AUC = {auc_roc:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        margin=dict(l=40, r=40, t=20, b=40),
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig

# Callback for prediction
@app.callback(
    Output("prediction-results", "children"),
    Input("predict-button", "n_clicks"),
    State("input-age", "value"),
    State("input-gender", "value"),
    State("input-tumor-type", "value"),
    State("input-cancer-stage", "value"),
    State("input-tumor-size", "value"),
    State("input-metastasis", "value"),
    State("input-treatment-type", "value"),
    State("input-chemo-sessions", "value"),
    State("input-radiation-sessions", "value"),
    State("input-smoking-status", "value"),
    State("input-alcohol-use", "value")
)
def update_prediction(n_clicks, age, gender, tumor_type, cancer_stage, tumor_size, 
                      metastasis, treatment_type, chemo_sessions, radiation_sessions,
                      smoking_status, alcohol_use):
    if n_clicks is None:
        return html.Div()
    
    try:
        # Map categorical values to numerical
        gender_male = 1 if gender == "Male" else 0
        metastasis_yes = 1 if metastasis == "Yes" else 0
        
        cancer_stage_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
        cancer_stage_encoded = cancer_stage_map[cancer_stage]
        
        smoking_status_map = {"Never": 0, "Former": 1, "Current": 2}
        smoking_status_encoded = smoking_status_map[smoking_status]
        
        alcohol_use_map = {"None": 0, "Moderate": 1, "Heavy": 2}
        alcohol_use_encoded = alcohol_use_map[alcohol_use]
        
        tumor_type_map = {"Lung": 0, "Colorectal": 1, "Breast": 2, "Liver": 3, "Stomach": 4, "Other": 5}
        tumor_type_encoded = tumor_type_map[tumor_type]
        
        treatment_type_map = {"Surgery": 0, "Chemotherapy": 1, "Radiation": 2, "Combined": 3, "Palliative": 4}
        treatment_type_encoded = treatment_type_map[treatment_type]
        
        # Calculate treatment complexity
        if treatment_type == "Palliative":
            treatment_complexity_encoded = 0  # Low
        elif treatment_type in ["Surgery", "Chemotherapy", "Radiation"]:
            treatment_complexity_encoded = 1 if chemo_sessions + radiation_sessions <= 15 else 2  # Medium or High
        else:  # Combined
            treatment_complexity_encoded = 1 if chemo_sessions + radiation_sessions <= 25 else 2  # Medium or High
        
        # Prepare features for scaling
        features_to_scale = [age, tumor_size, chemo_sessions, radiation_sessions, 0, 0]  # Last two are placeholders for FollowUpMonths and WaitingTimeDays
        
        # Scale features
        if 'scaler' in globals():
            scaled_features = scaler.transform([features_to_scale])[0]
        else:
            # If scaler is not available, use simple standardization
            scaled_features = [(features_to_scale[i] - [58.7, 4.2, 4.8, 8.6, 36.4, 18][i]) / [13.6, 2.8, 4.1, 10.3, 26.1, 9][i] for i in range(len(features_to_scale))]
        
        # Prepare input for prediction
        X_pred = [
            scaled_features[0],  # Age
            scaled_features[1],  # TumorSize
            scaled_features[2],  # ChemotherapySessions
            scaled_features[3],  # RadiationSessions
            cancer_stage_encoded,
            smoking_status_encoded,
            alcohol_use_encoded,
            gender_male,
            metastasis_yes,
            tumor_type_encoded,
            treatment_type_encoded,
            scaled_features[5],  # WaitingTimeDays
            treatment_complexity_encoded
        ]
        
        # Make prediction
        if 'logreg_model' in globals():
            # Use the trained model
            y_pred_proba = logreg_model.predict_proba([X_pred])[0][1]
        else:
            # If model is not available, use the coefficients from the JSON file
            intercept = model_coefficients["intercept"]
            coefficients = model_coefficients["coefficients"]
            
            # Match coefficients to input features
            coefficient_keys = [
                'Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions',
                'CancerStage_encoded', 'SmokingStatus_encoded', 'AlcoholUse_encoded',
                'Gender_Male', 'Metastasis_Yes', 'TumorType_encoded',
                'TreatmentType_encoded', 'WaitingTimeDays', 'TreatmentComplexity_encoded'
            ]
            
            # Calculate log-odds
            log_odds = intercept
            for i, key in enumerate(coefficient_keys):
                if key in coefficients:
                    log_odds += X_pred[i] * coefficients[key]
            
            # Convert to probability
            y_pred_proba = 1 / (1 + np.exp(-log_odds))
        
        # Determine predicted class
        y_pred = "Survived" if y_pred_proba >= 0.5 else "Deceased"
        
        # Prepare factor contributions for waterfall chart
        factor_contributions = []
        
        # Add most important positive and negative factors
        if cancer_stage_encoded >= 3:
            factor_contributions.append({"factor": "Advanced Cancer Stage", "impact": "negative"})
        else:
            factor_contributions.append({"factor": "Early Cancer Stage", "impact": "positive"})
        
        if metastasis_yes == 1:
            factor_contributions.append({"factor": "Metastasis Present", "impact": "negative"})
        else:
            factor_contributions.append({"factor": "No Metastasis", "impact": "positive"})
        
        if treatment_type in ["Combined", "Surgery"]:
            factor_contributions.append({"factor": f"{treatment_type} Treatment", "impact": "positive"})
        elif treatment_type == "Palliative":
            factor_contributions.append({"factor": "Palliative Care", "impact": "negative"})
        
        if age >= 70:
            factor_contributions.append({"factor": "Advanced Age", "impact": "negative"})
        elif age <= 45:
            factor_contributions.append({"factor": "Young Age", "impact": "positive"})
        
        if tumor_size >= 6:
            factor_contributions.append({"factor": "Large Tumor Size", "impact": "negative"})
        elif tumor_size <= 2:
            factor_contributions.append({"factor": "Small Tumor Size", "impact": "positive"})
        
        if smoking_status == "Current":
            factor_contributions.append({"factor": "Current Smoker", "impact": "negative"})
        
        if alcohol_use == "Heavy":
            factor_contributions.append({"factor": "Heavy Alcohol Use", "impact": "negative"})
        
        # Create prediction result display
        result_color = "rgb(46, 184, 46)" if y_pred == "Survived" else "rgb(220, 53, 69)"
        probability_percentage = int(y_pred_proba * 100)
        
        # Create factors list
        factors_list = html.Ul([
            html.Li([
                factor["factor"],
                html.Span(" ↑", className="positive-factor") if factor["impact"] == "positive" else html.Span(" ↓", className="negative-factor")
            ], className=f"{factor['impact']}-factor-item")
            for factor in factor_contributions
        ], className="factors-list")
        
        return html.Div([
            html.H3("Prediction Result", className="prediction-title"),
            
            html.Div([
                html.Div([
                    html.Div("Predicted Outcome:", className="prediction-label"),
                    html.Div(y_pred, className="prediction-value", style={"color": result_color})
                ], className="prediction-item"),
                
                html.Div([
                    html.Div("Probability:", className="prediction-label"),
                    html.Div([
                        html.Div(className="probability-bar-container", children=[
                            html.Div(
                                className="probability-bar",
                                style={"width": f"{probability_percentage}%", "background-color": result_color}
                            ),
                            html.Div(f"{probability_percentage}%", className="probability-text")
                        ])
                    ])
                ], className="prediction-item"),
            ], className="prediction-main"),
            
            html.Div([
                html.Div("Key Factors:", className="factors-title"),
                factors_list
            ], className="prediction-factors"),
            
            html.Div([
                html.P([
                    "This prediction is based on a logistic regression model trained on 10,000 cancer patient records. ",
                    "The model has an accuracy of 87.8% and an AUC-ROC of 0.89. ",
                    "The prediction should be used for informational purposes only and not as a substitute for professional medical advice."
                ], className="prediction-disclaimer")
            ], className="prediction-footer")
        ], className="prediction-container")
    
    except Exception as e:
        print(f"Error making prediction: {e}")
        return html.Div([
            html.H4("Prediction Error", className="error-title"),
            html.P(f"An error occurred while making the prediction: {str(e)}", className="error-message")
        ], className="error-container")

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)