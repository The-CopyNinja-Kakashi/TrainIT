import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import os

BASE_DIR = Path.cwd()
os.chdir(BASE_DIR)

@st.cache_resource
def load_components():
    try:
        metadata = joblib.load('./cluster_models/metadata.joblib')
        models = {}
        for cluster, model_info in metadata.items():
            models[int(cluster)] = joblib.load(f'./cluster_models/rf_cluster_{cluster}.joblib')
        
        df = pd.read_csv('./data/apy.csv')
        
        state_encoder = LabelEncoder().fit(df['State_Name'])
        crop_encoder = LabelEncoder().fit(df['Crop'])
        
        season_options = sorted(df['Season'].unique().tolist())
        
        kmeans = joblib.load('./cluster_models/kmeans_model.joblib')
        logyield_scaler = joblib.load('./cluster_models/logyield_scaler.joblib')
        
        return {
            'models': models,
            'kmeans': kmeans,
            'logyield_scaler': logyield_scaler,
            'state_encoder': state_encoder,
            'crop_encoder': crop_encoder,
            'state_options': list(state_encoder.classes_),
            'crop_options': list(crop_encoder.classes_),
            'season_options': season_options,
            'district_options': sorted(df['District_Name'].unique().tolist()),
            'avg_yields': pd.read_csv('./data/clustered_data3.csv')
        }
    
    except Exception as e:
        st.error(f"Error loading components: {str(e)}")
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
            
            avg_logyield = components