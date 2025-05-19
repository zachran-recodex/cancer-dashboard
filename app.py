import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pickle
import joblib
import os
import base64
import datetime

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

# Load preprocessed data and models - dengan penanganan error
print("Loading data and models...")
try:
    # Coba load versi cleaned terlebih dahulu
    df = pd.read_csv('data/preprocessed_cancer_data_clean.csv')
    print("Loaded clean preprocessed data")
except FileNotFoundError:
    try:
        # Jika tidak ada, load file original
        df = pd.read_csv('data/preprocessed_cancer_data.csv')
        print("Clean data not found, loaded original preprocessed data")
        
        # Cek dan tangani missing values
        missing_values = df.isnull().sum()
        missing_cols = missing_values[missing_values > 0]
        if len(missing_cols) > 0:
            print("Missing values detected in the following columns:")
            print(missing_cols)
            
            # Definisikan nilai default untuk setiap kolom
            default_values = {
                'CancerStage_encoded': 2,  # Stage II sebagai default
                'TreatmentType_encoded': 3,  # Combined treatment sebagai default
                'AlcoholUse_encoded': 1,  # Moderate sebagai default 
                'TumorType_encoded': 0,  # Lung sebagai default
                'Age': df['Age'].median() if 'Age' in df.columns and not df['Age'].isnull().all() else 60,
                'TumorSize': df['TumorSize'].median() if 'TumorSize' in df.columns and not df['TumorSize'].isnull().all() else 3.5,
                'ChemotherapySessions': df['ChemotherapySessions'].median() if 'ChemotherapySessions' in df.columns and not df['ChemotherapySessions'].isnull().all() else 6,
                'RadiationSessions': df['RadiationSessions'].median() if 'RadiationSessions' in df.columns and not df['RadiationSessions'].isnull().all() else 10,
                'SmokingStatus_encoded': 1,  # Former smoker sebagai default
                'Gender_Male': 0,  # Female sebagai default
                'Metastasis_Yes': 0,  # No metastasis sebagai default
                'Cluster': 0  # Cluster 0 sebagai default
            }

            # Isi nilai missing dengan default
            for col in df.columns:
                if col in default_values and df[col].isnull().sum() > 0:
                    print(f"Filling {df[col].isnull().sum()} missing values in {col} with {default_values[col]}")
                    df[col] = df[col].fillna(default_values[col])
            
            # Isi nilai missing yang tersisa
            df = df.fillna(0)
            print("All missing values filled")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        # Dataset dummy jika tidak bisa memuat data
        df = pd.DataFrame({
            'Age': np.random.normal(60, 10, 100),
            'TumorSize': np.random.normal(3.5, 1.5, 100),
            'ChemotherapySessions': np.random.randint(0, 12, 100),
            'RadiationSessions': np.random.randint(0, 20, 100),
            'CancerStage_encoded': np.random.randint(1, 5, 100),
            'SmokingStatus_encoded': np.random.randint(0, 3, 100),
            'AlcoholUse_encoded': np.random.randint(0, 3, 100),
            'Gender_Male': np.random.randint(0, 2, 100),
            'Metastasis_Yes': np.random.randint(0, 2, 100),
            'SurvivalStatus_Survived': np.random.randint(0, 2, 100),
            'Cluster': np.random.randint(0, 3, 100)
        })
        print("Created dummy dataset for demonstration")

# Load models with error handling
try:
    kmeans_model = joblib.load('model/kmeans_model.pkl')
    logistic_model = joblib.load('model/logistic_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    print("Successfully loaded all models")
    models_loaded = True
except Exception as e:
    print(f"Error loading models: {str(e)}")
    models_loaded = False

# Pastikan kolom 'Cluster' ada
if 'Cluster' not in df.columns:
    print("Cluster column not found, adding with random values")
    df['Cluster'] = np.random.randint(0, 3, len(df))

# Siapkan fitur visual untuk tab Overview
def get_age_distribution():
    if 'Age' in df.columns:
        fig = px.histogram(df, x='Age', color='Cluster', 
                         title='Distribusi Usia Pasien',
                         labels={'Age': 'Usia', 'count': 'Jumlah Pasien'})
        return fig
    return go.Figure()

def get_gender_distribution():
    if 'Gender_Male' in df.columns:
        gender_counts = df.groupby('Gender_Male').size().reset_index(name='count')
        gender_counts['Gender'] = gender_counts['Gender_Male'].map({0: 'Perempuan', 1: 'Laki-laki'})
        fig = px.pie(gender_counts, values='count', names='Gender', 
                    title='Distribusi Jenis Kelamin')
        return fig
    return go.Figure()

def get_tumor_type_distribution():
    if 'TumorType_encoded' in df.columns:
        tumor_mapping = {
            0: 'Paru-paru',
            1: 'Kolorektal',
            2: 'Payudara',
            3: 'Hati',
            4: 'Lambung',
            5: 'Lainnya'
        }
        df_tumor = df.copy()
        df_tumor['TumorType'] = df_tumor['TumorType_encoded'].map(tumor_mapping)
        fig = px.bar(df_tumor.groupby('TumorType').size().reset_index(name='count'), 
                    x='TumorType', y='count',
                    title='Distribusi Jenis Tumor',
                    labels={'TumorType': 'Jenis Tumor', 'count': 'Jumlah Pasien'})
        return fig
    return go.Figure()

# Cluster visualization function
def get_cluster_visualization():
    if 'Cluster' in df.columns:
        try:
            # Use PCA to visualize clusters in 2D
            features = ['Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 
                        'CancerStage_encoded', 'SmokingStatus_encoded', 'AlcoholUse_encoded', 
                        'Gender_Male', 'Metastasis_Yes']
            
            # Verify all features exist
            features_to_use = [f for f in features if f in df.columns]
            
            if len(features_to_use) >= 2:  # Need at least 2 features for PCA
                X = df[features_to_use].fillna(0)  # Handle any remaining NaNs
                pca = PCA(n_components=2)
                components = pca.fit_transform(X)
                
                # Create a DataFrame with PCA results
                df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
                df_pca['Cluster'] = df['Cluster']
                
                # Create the scatter plot
                fig = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                                title='Visualisasi Cluster dengan PCA',
                                labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'})
                return fig
            
        except Exception as e:
            print(f"Error creating cluster visualization: {str(e)}")
    
    # Fallback to empty figure
    fig = px.scatter(x=[0, 1, 2], y=[0, 1, 2], title="Tidak dapat membuat visualisasi cluster")
    return fig

def get_cluster_profiles():
    if 'Cluster' in df.columns:
        try:
            # Features to profile
            profile_features = ['Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 
                                'CancerStage_encoded', 'SurvivalStatus_Survived']
            
            # Select only features that exist in the DataFrame
            profile_features = [f for f in profile_features if f in df.columns]
            
            if len(profile_features) > 0:
                # Calculate mean values for each feature by cluster
                profiles = df.groupby('Cluster')[profile_features].mean().reset_index()
                
                # Melt the DataFrame for easier plotting
                profiles_melted = pd.melt(profiles, id_vars=['Cluster'], 
                                        value_vars=profile_features,
                                        var_name='Feature', value_name='Average Value')
                
                # Create the bar chart
                fig = px.bar(profiles_melted, x='Feature', y='Average Value', color='Cluster',
                            barmode='group', title='Profil Rata-rata Cluster')
                return fig
        
        except Exception as e:
            print(f"Error creating cluster profiles: {str(e)}")
    
    # Fallback to empty figure
    fig = px.bar(x=['No Data'], y=[0], title="Tidak dapat membuat profil cluster")
    return fig

# Confusion matrix visualization
def get_confusion_matrix():
    # Try to load the pre-computed confusion matrix plot
    try:
        img_path = 'results/confusion_matrix.png'
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')
            
            fig = html.Img(src=f'data:image/png;base64,{encoded_image}',
                          style={'width': '100%', 'max-width': '800px'})
            return fig
    except Exception as e:
        print(f"Error loading confusion matrix image: {str(e)}")
    
    # Fallback to empty div
    return html.Div("Gambar confusion matrix tidak tersedia. Jalankan test_models.py terlebih dahulu.")

# ROC curve visualization
def get_roc_curve():
    # Try to load the pre-computed ROC curve plot
    try:
        img_path = 'results/roc_curve.png'
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')
            
            fig = html.Img(src=f'data:image/png;base64,{encoded_image}',
                          style={'width': '100%', 'max-width': '800px'})
            return fig
    except Exception as e:
        print(f"Error loading ROC curve image: {str(e)}")
    
    # Fallback to empty div
    return html.Div("Gambar ROC curve tidak tersedia. Jalankan test_models.py terlebih dahulu.")

# Feature importance visualization
def get_feature_importance():
    # Try to load the pre-computed feature importance plot
    try:
        img_path = 'results/feature_importance.png'
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')
            
            fig = html.Img(src=f'data:image/png;base64,{encoded_image}',
                          style={'width': '100%', 'max-width': '800px'})
            return fig
    except Exception as e:
        print(f"Error loading feature importance image: {str(e)}")
    
    # Fallback to empty div
    return html.Div("Gambar feature importance tidak tersedia. Jalankan test_models.py terlebih dahulu.")

# Define app layout
app.layout = html.Div([
    html.H1("Dashboard Analisis Pasien Kanker", style={'textAlign': 'center'}),
    
    # Warning if models not loaded
    html.Div([
        html.Div([
            html.H4("⚠️ Peringatan: Model tidak berhasil dimuat", style={'color': 'red'}),
            html.P("Beberapa fitur dashboard mungkin tidak berfungsi. Silakan jalankan save_models.py terlebih dahulu.")
        ], style={'padding': '10px', 'border': '1px solid red', 'borderRadius': '5px', 'backgroundColor': '#ffeeee'})
    ] if not models_loaded else [], id='model-warning'),
    
    # Tabs for different sections
    dcc.Tabs([
        # Tab 1: Overview
        dcc.Tab(label='Overview', children=[
            html.Div([
                html.H3("Distribusi Demografis Pasien"),
                html.Div([
                    dcc.Graph(id='age-distribution', figure=get_age_distribution()),
                    dcc.Graph(id='gender-distribution', figure=get_gender_distribution()),
                    dcc.Graph(id='tumor-distribution', figure=get_tumor_type_distribution())
                ])
            ])
        ]),
        
        # Tab 2: Exploratory Analysis
        dcc.Tab(label='Exploratory Analysis', children=[
            html.Div([
                html.H3("Analisis Eksplorasi Data"),
                html.Div([
                    html.Div([
                        html.Label('Pilih Variabel X:'),
                        dcc.Dropdown(
                            id='x-variable',
                            options=[{'label': col, 'value': col} for col in df.columns],
                            value='Age' if 'Age' in df.columns else df.columns[0]
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
                    html.Div([
                        html.Label('Pilih Variabel Y:'),
                        dcc.Dropdown(
                            id='y-variable',
                            options=[{'label': col, 'value': col} for col in df.columns],
                            value='TumorSize' if 'TumorSize' in df.columns else df.columns[0]
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
                    html.Div([
                        html.Label('Pilih Jenis Plot:'),
                        dcc.Dropdown(
                            id='plot-type',
                            options=[
                                {'label': 'Scatter Plot', 'value': 'scatter'},
                                {'label': 'Bar Plot', 'value': 'bar'},
                                {'label': 'Box Plot', 'value': 'box'},
                                {'label': 'Histogram', 'value': 'histogram'}
                            ],
                            value='scatter'
                        )
                    ], style={'width': '30%', 'display': 'inline-block'})
                ]),
                dcc.Graph(id='exploratory-plot')
            ])
        ]),
        
        # Tab 3: Clustering Results
        dcc.Tab(label='Clustering Results', children=[
            html.Div([
                html.H3("Hasil K-means Clustering"),
                dcc.Graph(id='cluster-visualization', figure=get_cluster_visualization()),
                dcc.Graph(id='cluster-profiles', figure=get_cluster_profiles()),
                html.Div([
                    html.Label('Pilih Cluster:'),
                    dcc.Dropdown(
                        id='cluster-selector',
                        options=[
                            {'label': 'Cluster 0: Young Survivors', 'value': 0},
                            {'label': 'Cluster 1: Mid-stage Patients', 'value': 1},
                            {'label': 'Cluster 2: Advanced Cases', 'value': 2}
                        ],
                        value=0
                    )
                ], style={'width': '50%'}),
                dcc.Graph(id='cluster-details')
            ])
        ]),
        
        # Tab 4: Survival Prediction
        dcc.Tab(label='Survival Prediction', children=[
            html.Div([
                html.H3("Prediksi Status Kelangsungan Hidup"),
                
                # Notification about synthetic data if needed
                html.Div([
                    html.Div([
                        html.P("⚠️ Catatan: Model ini menggunakan data sintetis untuk menyeimbangkan kelas. Hasil prediksi hanya untuk tujuan demonstrasi.", 
                              style={'color': '#856404', 'marginBottom': '0'})
                    ], style={'padding': '10px', 'backgroundColor': '#fff3cd', 'borderRadius': '5px', 'marginBottom': '20px'})
                ] if not models_loaded else []),
                
                html.Div([
                    html.Div([
                        html.Label('Usia:'),
                        dcc.Input(id='input-age', type='number', value=60),
                        html.Label('Jenis Kelamin:'),
                        dcc.Dropdown(
                            id='input-gender',
                            options=[
                                {'label': 'Laki-laki', 'value': 1},
                                {'label': 'Perempuan', 'value': 0}
                            ],
                            value=1
                        ),
                        html.Label('Jenis Tumor:'),
                        dcc.Dropdown(
                            id='input-tumor-type',
                            options=[
                                {'label': 'Paru-paru', 'value': 0},
                                {'label': 'Kolorektal', 'value': 1},
                                {'label': 'Payudara', 'value': 2},
                                {'label': 'Hati', 'value': 3},
                                {'label': 'Lambung', 'value': 4},
                                {'label': 'Lainnya', 'value': 5}
                            ],
                            value=0
                        ),
                        html.Label('Stadium Kanker:'),
                        dcc.Dropdown(
                            id='input-cancer-stage',
                            options=[
                                {'label': 'Stadium I', 'value': 1},
                                {'label': 'Stadium II', 'value': 2},
                                {'label': 'Stadium III', 'value': 3},
                                {'label': 'Stadium IV', 'value': 4}
                            ],
                            value=2
                        )
                    ], style={'width': '30%', 'float': 'left'}),
                    html.Div([
                        html.Label('Ukuran Tumor (cm):'),
                        dcc.Input(id='input-tumor-size', type='number', value=4.0),
                        html.Label('Metastasis:'),
                        dcc.Dropdown(
                            id='input-metastasis',
                            options=[
                                {'label': 'Ya', 'value': 1},
                                {'label': 'Tidak', 'value': 0}
                            ],
                            value=0
                        ),
                        html.Label('Jenis Pengobatan:'),
                        dcc.Dropdown(
                            id='input-treatment-type',
                            options=[
                                {'label': 'Operasi', 'value': 0},
                                {'label': 'Kemoterapi', 'value': 1},
                                {'label': 'Radiasi', 'value': 2},
                                {'label': 'Kombinasi', 'value': 3},
                                {'label': 'Paliatif', 'value': 4}
                            ],
                            value=3
                        ),
                        html.Label('Status Merokok:'),
                        dcc.Dropdown(
                            id='input-smoking-status',
                            options=[
                                {'label': 'Tidak Pernah', 'value': 0},
                                {'label': 'Mantan Perokok', 'value': 1},
                                {'label': 'Perokok Aktif', 'value': 2}
                            ],
                            value=1
                        )
                    ], style={'width': '30%', 'float': 'left', 'marginLeft': '5%'}),
                    html.Div([
                        html.Label('Konsumsi Alkohol:'),
                        dcc.Dropdown(
                            id='input-alcohol-use',
                            options=[
                                {'label': 'Tidak', 'value': 0},
                                {'label': 'Moderat', 'value': 1},
                                {'label': 'Berat', 'value': 2}
                            ],
                            value=1
                        ),
                        html.Label('Sesi Kemoterapi:'),
                        dcc.Input(id='input-chemo-sessions', type='number', value=5),
                        html.Label('Sesi Radiasi:'),
                        dcc.Input(id='input-radiation-sessions', type='number', value=10),
                        html.Br(),
                        html.Button('Prediksi', id='predict-button', n_clicks=0,
                                   style={'backgroundColor': '#4CAF50', 'color': 'white', 
                                          'padding': '10px 15px', 'borderRadius': '5px',
                                          'border': 'none', 'marginTop': '20px', 'cursor': 'pointer'})
                    ], style={'width': '30%', 'float': 'left', 'marginLeft': '5%'})
                ], style={'overflow': 'hidden'}),
                html.Div(id='prediction-output', style={'marginTop': '50px'})
            ])
        ]),
        
        # Tab 5: Model Performance
        dcc.Tab(label='Model Performance', children=[
            html.Div([
                html.H3("Performa Model Prediktif"),
                
                # Instructions box
                html.Div([
                    html.P("Untuk menampilkan visualisasi performa model, jalankan script test_models.py terlebih dahulu."),
                    html.P("File-file visualisasi akan disimpan di direktori 'results'.")
                ], style={'padding': '10px', 'backgroundColor': '#e9ecef', 'borderRadius': '5px', 'marginBottom': '20px'}),
                
                html.Div([
                    html.H4("Confusion Matrix", style={'marginTop': '30px'}),
                    html.Div(id='confusion-matrix', children=get_confusion_matrix()),
                    
                    html.H4("ROC Curve", style={'marginTop': '30px'}),
                    html.Div(id='roc-curve', children=get_roc_curve()),
                    
                    html.H4("Feature Importance", style={'marginTop': '30px'}),
                    html.Div(id='feature-importance', children=get_feature_importance())
                ])
            ])
        ])
    ]),
    
    # Footer
    html.Div([
        html.Hr(),
        html.P(f"Dashboard Analisis Pasien Kanker | Data terakhir diupdate: {datetime.datetime.now().strftime('%Y-%m-%d')}"),
        html.P("Catatan: Dashboard ini dibuat untuk tujuan pembelajaran dan demonstrasi.")
    ], style={'marginTop': '50px', 'textAlign': 'center', 'color': '#6c757d'})
])

# Callback for exploratory plot
@app.callback(
    Output('exploratory-plot', 'figure'),
    [Input('x-variable', 'value'),
     Input('y-variable', 'value'),
     Input('plot-type', 'value')]
)
def update_exploratory_plot(x_var, y_var, plot_type):
    try:
        # Ensure both variables exist in the dataframe
        if x_var not in df.columns or y_var not in df.columns:
            return px.scatter(title="Error: One or both selected variables not found in dataset")
        
        # Handle missing values
        temp_df = df.copy()
        if temp_df[x_var].isnull().any() or temp_df[y_var].isnull().any():
            temp_df = temp_df.fillna(0)
        
        # Create plot based on type
        if plot_type == 'scatter':
            fig = px.scatter(temp_df, x=x_var, y=y_var, color='Cluster',
                           title=f'Scatter Plot: {x_var} vs {y_var}')
        elif plot_type == 'bar':
            if x_var == y_var:
                fig = px.histogram(temp_df, x=x_var, color='Cluster',
                                 title=f'Histogram: {x_var}')
            else:
                # Ensure x_var is categorical or binned
                if temp_df[x_var].nunique() > 10 and pd.api.types.is_numeric_dtype(temp_df[x_var]):
                    # Bin the variable if it has too many unique values
                    temp_df[f'{x_var}_binned'] = pd.cut(temp_df[x_var], bins=10)
                    group_var = f'{x_var}_binned'
                else:
                    group_var = x_var
                
                grouped = temp_df.groupby([group_var, 'Cluster'])[y_var].mean().reset_index()
                fig = px.bar(grouped, x=group_var, y=y_var, color='Cluster',
                           title=f'Bar Plot: {x_var} vs {y_var}')
        elif plot_type == 'box':
            fig = px.box(temp_df, x=x_var, y=y_var, color='Cluster',
                       title=f'Box Plot: {x_var} vs {y_var}')
        else:  # histogram
            fig = px.histogram(temp_df, x=x_var, color='Cluster',
                             title=f'Histogram: {x_var}')
        
        # Layout adjustments
        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray')
        )
        
        return fig
    
    except Exception as e:
        print(f"Error in exploratory plot: {str(e)}")
        return px.scatter(title=f"Error creating plot: {str(e)}")

# Callback for cluster details
@app.callback(
    Output('cluster-details', 'figure'),
    [Input('cluster-selector', 'value')]
)
def update_cluster_details(selected_cluster):
    try:
        # Filter for the selected cluster
        if 'Cluster' not in df.columns:
            return px.scatter(title="Error: Cluster column not found in dataset")
        
        cluster_df = df[df['Cluster'] == selected_cluster]
        
        if len(cluster_df) == 0:
            return px.scatter(title=f"No data for Cluster {selected_cluster}")
        
        # Basic cluster statistics
        # Survival rate by tumor type
        if 'TumorType_encoded' in cluster_df.columns and 'SurvivalStatus_Survived' in cluster_df.columns:
            tumor_mapping = {
                0: 'Paru-paru',
                1: 'Kolorektal',
                2: 'Payudara',
                3: 'Hati',
                4: 'Lambung',
                5: 'Lainnya'
            }
            cluster_df['TumorType'] = cluster_df['TumorType_encoded'].map(tumor_mapping)
            
            survival_by_tumor = cluster_df.groupby('TumorType')['SurvivalStatus_Survived'].mean().reset_index()
            
            fig = px.bar(survival_by_tumor, x='TumorType', y='SurvivalStatus_Survived',
                       title=f'Tingkat Kelangsungan Hidup berdasarkan Jenis Tumor untuk Cluster {selected_cluster}',
                       labels={'TumorType': 'Jenis Tumor', 
                              'SurvivalStatus_Survived': 'Tingkat Kelangsungan Hidup'})
            
            # Layout adjustments
            fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(gridcolor='lightgray'),
                yaxis=dict(gridcolor='lightgray', range=[0, 1])
            )
            
            return fig
        else:
            # Fallback to age distribution if tumor type not available
            if 'Age' in cluster_df.columns:
                fig = px.histogram(cluster_df, x='Age',
                                 title=f'Distribusi Usia untuk Cluster {selected_cluster}')
                return fig
            
        # Fallback when neither option is available
        return px.scatter(title=f"Detail untuk Cluster {selected_cluster} tidak tersedia")
    
    except Exception as e:
        print(f"Error in cluster details: {str(e)}")
        return px.scatter(title=f"Error menampilkan detail cluster: {str(e)}")

# Callback for prediction
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('input-age', 'value'),
     State('input-gender', 'value'),
     State('input-tumor-type', 'value'),
     State('input-cancer-stage', 'value'),
     State('input-tumor-size', 'value'),
     State('input-metastasis', 'value'),
     State('input-treatment-type', 'value'),
     State('input-smoking-status', 'value'),
     State('input-alcohol-use', 'value'),
     State('input-chemo-sessions', 'value'),
     State('input-radiation-sessions', 'value')]
)
def predict_survival(n_clicks, age, gender, tumor_type, cancer_stage, tumor_size,
                    metastasis, treatment_type, smoking_status, alcohol_use,
                    chemo_sessions, radiation_sessions):
    # Don't update on initial load
    if n_clicks == 0:
        return html.Div()
    
    # Check if models are loaded
    if not models_loaded:
        return html.Div([
            html.H4("Model tidak tersedia", style={'color': 'red'}),
            html.P("Tidak dapat melakukan prediksi karena model tidak berhasil dimuat. Jalankan save_models.py terlebih dahulu.")
        ], style={'padding': '20px', 'backgroundColor': '#f8d7da', 'borderRadius': '5px'})
    
    try:
        # Prepare input data more explicitly
        # Create feature vector with the correct features and order
        features = ['Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 
                    'CancerStage_encoded', 'SmokingStatus_encoded', 'AlcoholUse_encoded', 
                    'Gender_Male', 'Metastasis_Yes']
        
        # Initialize with zeros
        input_data = np.zeros(len(features))
        
        # Fill in the values
        feature_values = {
            'Age': age,
            'TumorSize': tumor_size,
            'ChemotherapySessions': chemo_sessions,
            'RadiationSessions': radiation_sessions,
            'CancerStage_encoded': cancer_stage,
            'SmokingStatus_encoded': smoking_status,
            'AlcoholUse_encoded': alcohol_use,
            'Gender_Male': gender,
            'Metastasis_Yes': metastasis
        }
        
        # Populate the input data array with the correct values in the correct positions
        for i, feature in enumerate(features):
            input_data[i] = feature_values.get(feature, 0)
        
        # Scale input data
        input_scaled = scaler.transform([input_data])
        
        # Make prediction
        survival_prob = logistic_model.predict_proba(input_scaled)[0, 1]
        prediction = "Survived" if survival_prob >= 0.5 else "Deceased"
        prediction_id = "survived" if survival_prob >= 0.5 else "deceased"
        
        # Create output display
        return html.Div([
            html.Div([
                html.H4(f"Prediksi Status: ", style={'display': 'inline-block', 'marginRight': '10px'}),
                html.H4(prediction, style={'display': 'inline-block', 
                                           'color': 'green' if prediction == "Survived" else 'red'})
            ]),
            html.H5(f"Probabilitas Bertahan Hidup: {survival_prob:.2f}"),
            dcc.Graph(
                figure=go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=survival_prob * 100,
                        title={'text': "Probabilitas Kelangsungan Hidup (%)"},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': "darkgreen"},
                               'steps': [
                                   {'range': [0, 25], 'color': "red"},
                                   {'range': [25, 50], 'color': "orange"},
                                   {'range': [50, 75], 'color': "yellow"},
                                   {'range': [75, 100], 'color': "lightgreen"}
                               ],
                               'threshold': {
                                   'line': {'color': "black", 'width': 4},
                                   'thickness': 0.75,
                                   'value': survival_prob * 100
                               }}
                    )
                )
            ),
            html.Div([
                html.H5("Faktor-faktor Penting yang Mempengaruhi Prediksi:"),
                html.Ul([
                    html.Li(f"Usia: {age} tahun", className=prediction_id),
                    html.Li(f"Stadium Kanker: {cancer_stage}", 
                           className=prediction_id if cancer_stage < 3 else "risk-factor"),
                    html.Li(f"Metastasis: {'Ya' if metastasis == 1 else 'Tidak'}", 
                           className=prediction_id if metastasis == 0 else "risk-factor"),
                    html.Li(f"Ukuran Tumor: {tumor_size} cm", 
                           className=prediction_id if tumor_size < 5 else "risk-factor"),
                ])
            ], style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
        ])
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return html.Div([
            html.H4("Error dalam Prediksi", style={'color': 'red'}),
            html.P(f"Terjadi kesalahan: {str(e)}")
        ], style={'padding': '20px', 'backgroundColor': '#f8d7da', 'borderRadius': '5px'})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)