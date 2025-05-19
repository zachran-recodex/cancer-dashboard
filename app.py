import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import joblib

# Load preprocessed data and models
df = pd.read_csv('data/preprocessed_cancer_data.csv')
kmeans_model = joblib.load('model/kmeans_model.pkl')
logistic_model = joblib.load('model/logistic_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

# Define app layout
app.layout = html.Div([
    html.H1("Dashboard Analisis Pasien Kanker", style={'textAlign': 'center'}),
    
    # Tabs for different sections
    dcc.Tabs([
        # Tab 1: Overview
        dcc.Tab(label='Overview', children=[
            html.Div([
                html.H3("Distribusi Demografis Pasien"),
                dcc.Graph(id='age-distribution'),
                dcc.Graph(id='gender-distribution'),
                dcc.Graph(id='province-distribution')
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
                            value='Age'
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
                    html.Div([
                        html.Label('Pilih Variabel Y:'),
                        dcc.Dropdown(
                            id='y-variable',
                            options=[{'label': col, 'value': col} for col in df.columns],
                            value='TumorSize'
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
                dcc.Graph(id='cluster-visualization'),
                dcc.Graph(id='cluster-profiles'),
                html.Div([
                    html.Label('Pilih Cluster:'),
                    dcc.Dropdown(
                        id='cluster-selector',
                        options=[
                            {'label': 'Cluster 1: Young Survivors', 'value': 0},
                            {'label': 'Cluster 2: Mid-stage Patients', 'value': 1},
                            {'label': 'Cluster 3: Advanced Cases', 'value': 2}
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
                        html.Button('Prediksi', id='predict-button', n_clicks=0)
                    ], style={'width': '30%', 'float': 'left', 'marginLeft': '5%'})
                ], style={'overflow': 'hidden'}),
                html.Div(id='prediction-output', style={'marginTop': '50px'})
            ])
        ]),
        
        # Tab 5: Model Performance
        dcc.Tab(label='Model Performance', children=[
            html.Div([
                html.H3("Performa Model Prediktif"),
                dcc.Graph(id='confusion-matrix'),
                dcc.Graph(id='roc-curve'),
                dcc.Graph(id='feature-importance')
            ])
        ])
    ])
])

# Callback for exploratory plot
@app.callback(
    Output('exploratory-plot', 'figure'),
    [Input('x-variable', 'value'),
     Input('y-variable', 'value'),
     Input('plot-type', 'value')]
)
def update_exploratory_plot(x_var, y_var, plot_type):
    if plot_type == 'scatter':
        fig = px.scatter(df, x=x_var, y=y_var, color='Cluster',
                         title=f'Scatter Plot: {x_var} vs {y_var}')
    elif plot_type == 'bar':
        if x_var == y_var:
            fig = px.histogram(df, x=x_var, color='Cluster',
                               title=f'Histogram: {x_var}')
        else:
            fig = px.bar(df.groupby([x_var, 'Cluster'])[y_var].mean().reset_index(),
                         x=x_var, y=y_var, color='Cluster',
                         title=f'Bar Plot: {x_var} vs {y_var}')
    elif plot_type == 'box':
        fig = px.box(df, x=x_var, y=y_var, color='Cluster',
                     title=f'Box Plot: {x_var} vs {y_var}')
    else:  # histogram
        fig = px.histogram(df, x=x_var, color='Cluster',
                           title=f'Histogram: {x_var}')
    
    return fig

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
    if n_clicks == 0:
        return html.Div()
    
    # Prepare input data (simplified for illustration)
    input_data = np.zeros(15)  # Assume 15 features after preprocessing
    input_data[0] = age
    input_data[1] = tumor_size
    input_data[2] = chemo_sessions
    input_data[3] = radiation_sessions
    input_data[4] = cancer_stage
    input_data[5] = smoking_status
    input_data[6] = alcohol_use
    input_data[7] = gender
    input_data[8] = metastasis
    
    # Scale input data
    input_scaled = scaler.transform([input_data])
    
    # Make prediction
    survival_prob = logistic_model.predict_proba(input_scaled)[0, 1]
    prediction = "Survived" if survival_prob >= 0.5 else "Deceased"
    
    # Create output display
    return html.Div([
        html.H4(f"Prediksi Status: {prediction}"),
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
        )
    ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)