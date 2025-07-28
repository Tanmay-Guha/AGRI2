import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Generate synthetic crop data
np.random.seed(42)
df = pd.DataFrame({
    'temperature': np.random.uniform(15, 40, 100),
    'rainfall': np.random.uniform(50, 300, 100),
    'humidity': np.random.uniform(30, 90, 100),
    'N': np.random.randint(0, 140, 100),
    'P': np.random.randint(5, 120, 100),
    'K': np.random.randint(5, 200, 100),
    'ph': np.random.uniform(5.5, 8.0, 100),
    'organic_content': np.random.uniform(0.5, 3.5, 100),
    'sunlight_hours': np.random.uniform(4, 12, 100),
    'crop_type': np.random.choice(['wheat', 'rice', 'corn', 'barley'], 100)
})
df['yield'] = (
    df['temperature'] * 0.4 +
    df['rainfall'] * 0.2 +
    df['humidity'] * 0.3 +
    df['N'] * 0.05 -
    df['ph'] * 0.1 +
    df['organic_content'] * 2 +
    df['sunlight_hours'] * 0.8 +
    np.random.normal(0, 10, 100)
)

# Preprocessing pipeline
categorical = ['crop_type']
numeric = ['temperature', 'rainfall', 'humidity', 'N', 'P', 'K', 'ph', 'organic_content', 'sunlight_hours']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(), categorical)
])

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

X = df[numeric + categorical]
y = df['yield']
pipeline.fit(X, y)
joblib.dump(pipeline, 'crop_model.pkl')  # Save for reuse

model = pipeline

# Dash app
app = dash.Dash(__name__)
app.title = "Crop Yield Prediction"

app.layout = html.Div([
    html.H1("🌾 Crop Yield Predictor Dashboard", style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            html.Label("🌡️ Temperature (°C)"),
            dcc.Slider(id='temperature', min=10, max=45, step=0.5, value=25),
            html.Label("🌧️ Rainfall (mm)"),
            dcc.Slider(id='rainfall', min=0, max=400, step=5, value=150),
            html.Label("💧 Humidity (%)"),
            dcc.Slider(id='humidity', min=10, max=100, step=1, value=60),
            html.Label("🧪 Nitrogen (N)"),
            dcc.Slider(id='N', min=0, max=140, step=1, value=50),
            html.Label("🧪 Phosphorus (P)"),
            dcc.Slider(id='P', min=0, max=140, step=1, value=50),
            html.Label("🧪 Potassium (K)"),
            dcc.Slider(id='K', min=0, max=200, step=1, value=50),
            html.Label("⚗️ Soil pH"),
            dcc.Slider(id='ph', min=4, max=9, step=0.1, value=6.5),
            html.Label("🌱 Organic Content (%)"),
            dcc.Slider(id='organic_content', min=0, max=5, step=0.1, value=2),
            html.Label("☀️ Sunlight Hours"),
            dcc.Slider(id='sunlight_hours', min=0, max=15, step=0.5, value=8),
            html.Label("🌾 Crop Type"),
            dcc.Dropdown(
                id='crop_type',
                options=[{'label': crop, 'value': crop} for crop in df['crop_type'].unique()],
                value='wheat'
            )
        ], style={'columnCount': 2, 'padding': '20px'})
    ]),
    html.H2(id='prediction-output', style={'textAlign': 'center', 'color': 'green', 'marginTop': '20px'})
])

@app.callback(
    Output('prediction-output', 'children'),
    Input('temperature', 'value'),
    Input('rainfall', 'value'),
    Input('humidity', 'value'),
    Input('N', 'value'),
    Input('P', 'value'),
    Input('K', 'value'),
    Input('ph', 'value'),
    Input('organic_content', 'value'),
    Input('sunlight_hours', 'value'),
    Input('crop_type', 'value')
)
def predict_yield(temp, rain, hum, N, P, K, ph, org, sun, crop):
    try:
        input_df = pd.DataFrame([{
            'temperature': temp,
            'rainfall': rain,
            'humidity': hum,
            'N': N,
            'P': P,
            'K': K,
            'ph': ph,
            'organic_content': org,
            'sunlight_hours': sun,
            'crop_type': crop
        }])
        prediction = model.predict(input_df)[0]
        return f"✅ Predicted Yield: {prediction:.2f} units"
    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
