import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import os

# Path setup - files are in main directory
BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_components():
    try:
        # Load models from cluster_models folder
        metadata = joblib.load(BASE_DIR / 'cluster_models/metadata.joblib')
        models = {}
        for cluster, model_info in metadata.items():
            models[int(cluster)] = joblib.load(BASE_DIR / f'cluster_models/rf_cluster_{cluster}.joblib')
        
        # Load data files from main directory
        df = pd.read_csv(BASE_DIR / 'apy.csv')
        clustered_df = pd.read_csv(BASE_DIR / 'clustered_data3.csv')
        
        # Create encoders
        state_encoder = LabelEncoder().fit(df['State_Name'])
        crop_encoder = LabelEncoder().fit(df['Crop'])
        
        # Load clustering components
        kmeans = joblib.load(BASE_DIR / 'cluster_models/kmeans_model.joblib')
        logyield_scaler = joblib.load(BASE_DIR / 'cluster_models/logyield_scaler.joblib')
        
        return {
            'models': models,
            'kmeans': kmeans,
            'logyield_scaler': logyield_scaler,
            'state_encoder': state_encoder,
            'crop_encoder': crop_encoder,
            'state_options': list(state_encoder.classes_),
            'crop_options': list(crop_encoder.classes_),
            'season_options': sorted(df['Season'].unique().tolist()),
            'district_options': sorted(df['District_Name'].unique().tolist()),
            'avg_yields': clustered_df
        }
    
    except Exception as e:
        st.error(f"Error loading components: {str(e)}")
        st.error(f"Current working directory: {os.getcwd()}")
        st.error(f"BASE_DIR contents: {list(BASE_DIR.glob('*'))}")
        return None

def main():
    st.title("🌾 Agricultural Production Predictor")
    
    components = load_components()
    if components is None:
        return
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            state = st.selectbox("State", components['state_options'])
            district = st.selectbox("District", components['district_options'])
            
        with col2:
            crop = st.selectbox("Crop", components['crop_options'])
            season = st.selectbox("Season", components['season_options'])
            area = st.number_input("Area (hectares)", min_value=0.1, value=1.0, step=0.1)
        
        submitted = st.form_submit_button("Predict Production")

    if submitted:
        try:
            state_encoded = components['state_encoder'].transform([state])[0]
            crop_encoded = components['crop_encoder'].transform([crop])[0]
            
            avg_logyield = components['avg_yields'][
                components['avg_yields']['Crop'] == crop_encoded
            ]['LogYield'].mean()
            
            scaled_logyield = components['logyield_scaler'].transform(
                [[avg_logyield]]
            )[0][0]
            
            season_dummy = {s: 1 if s == season else 0 
                          for s in components['season_options']}
            
            cluster_input = pd.DataFrame([{
                'scaled_LogYield': scaled_logyield,
                **season_dummy
            }])
            
            cluster = components['kmeans'].predict(cluster_input)[0]
            
            input_data = {
                'Area': area,
                'State_Name': state_encoded,
                'Crop': crop_encoded,
                **season_dummy
            }
            input_df = pd.DataFrame([input_data])
            
            model_data = components['models'][cluster]
            X = input_df[model_data['features']]
            X_scaled = model_data['scaler'].transform(X)
            
            prediction = model_data['model'].predict(X_scaled)[0]
            
            st.success(f"**Predicted Production:** {prediction:,.2f} units")
            st.info(f"""
            - **Cluster:** {cluster}
            - **Model MAE:** {model_data['Mae']:.2f}
            - **Model R²:** {model_data['r2']:.2f}
            """)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
