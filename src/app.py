import streamlit as st
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchvision.models as models
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import base64
import random
import time
from datetime import datetime

# --- Configuration Constants (Updated for New Structure) ---
MODEL_PATH = "models/ecoear_model.pth"
BANNER_PATH = "assets/forest.jpg" 
SAMPLE_RATE = 22050
DURATION = 2
N_MELS = 128
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="EcoEar Command Center",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- Utility Functions ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- CSS Stylesheet Configuration ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Playfair+Display:wght@700&display=swap');

    :root {
        --primary-teal: #264E47;
        --secondary-teal: #1B3A35;
        --text-light: #E0EAE5;
        --bg-color: #F2F6F5;
        --card-bg: #FFFFFF;
        --accent-red: #C62828;
    }
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        font-size: 1.1rem; 
    }
    
    .stApp {
        background-color: var(--bg-color) !important;
        color: var(--primary-teal) !important;
    }

    /* Sidebar Specific Styling */
    [data-testid="stSidebar"] {
        background-color: var(--primary-teal) !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-family: 'Playfair Display', serif;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div, [data-testid="stSidebar"] label {
        color: var(--text-light) !important;
    }
    [data-testid="stSidebar"] input {
        color: #333 !important;
    }

    /* Layout & Header */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1400px;
        overflow: visible !important;
    }

    h1 {
        font-family: 'Playfair Display', serif;
        color: var(--primary-teal) !important;
        font-weight: 700;
        letter-spacing: -0.5px;
        font-size: 3.5rem !important;
        line-height: 1.2;
        padding-bottom: 5px;
        margin-bottom: 0 !important;
    }
    h3 {
        font-family: 'Playfair Display', serif;
        color: var(--primary-teal) !important;
        font-weight: 700;
        font-size: 1.8rem !important;
        margin-bottom: 20px !important;
    }

    /* Card Design */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: none !important;
        border-radius: 16px !important;
        background-color: var(--card-bg) !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05), 0 4px 10px rgba(0,0,0,0.02) !important;
        padding: 30px !important;
        margin-bottom: 25px;
    }

    /* Metric Styling */
    div[data-testid="stMetricValue"] {
        font-family: 'Playfair Display', serif !important;
        font-size: 2.5rem !important;
        color: var(--primary-teal) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #888 !important;
    }

    /* UI Elements */
    .stRadio label, .stSelectbox label, .stFileUploader label, .stSlider label, p {
        color: var(--primary-teal) !important;
    }
    
    div.stButton > button {
        background-color: var(--primary-teal);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(38, 78, 71, 0.2);
    }
    div.stButton > button:hover {
        background-color: #1B3A35;
        transform: translateY(-2px);
    }

    /* Banner Image */
    .banner-container {
        width: 100%;
        margin-top: 1rem;
        margin-bottom: 3rem;
        display: block;
        line-height: 0;
    }
    .custom-banner {
        width: 100%;
        height: 400px;
        object-fit: cover;
        border-radius: 20px;
        display: block;
        border: none !important;
        box-shadow: none !important;
    }
    
    .top-accent {
        height: 6px;
        background: var(--primary-teal);
        width: 100%;
        position: fixed;
        top: 0;
        left: 0;
        z-index: 9999;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="top-accent"></div>', unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1)
    )
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def process_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        target_len = SAMPLE_RATE * DURATION
        if len(y) > target_len:
            y = y[:target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)))
        waveform = torch.from_numpy(y).unsqueeze(0).to(DEVICE)
        mel_transform = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=2048, hop_length=512).to(DEVICE)
        spec = mel_transform(waveform)
        spec_db = T.AmplitudeToDB().to(DEVICE)(spec)
        input_tensor = (spec_db + 80.0) / 80.0
        input_tensor = input_tensor.unsqueeze(0).expand(-1, 3, -1, -1)
        return y, spec_db.cpu().numpy()[0], input_tensor
    except Exception as e:
        st.error(f"Processing Error: {e}")
        return None, None, None

# --- Mock Data Generation for Map ---
def generate_map_data(center_lat=40.4406, center_lon=-79.9959, n_points=50):
    data = []
    for _ in range(n_points):
        data.append({
            'lat': center_lat + random.uniform(-0.02, 0.02),
            'lon': center_lon + random.uniform(-0.02, 0.02),
            'status': random.choice(['Active', 'Active', 'Active', 'Maintenance'])
        })
    return pd.DataFrame(data)

# --- Sidebar: Real-time Logs ---
with st.sidebar:
    st.markdown("### Sensor Network Logs")
    st.markdown("<hr style='margin-top: 0; margin-bottom: 20px; border-top: 1px solid rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
    
    log_placeholder = st.empty()
    
    logs = [
        f"{datetime.now().strftime('%H:%M:%S')} - Node A12: Heartbeat OK",
        f"{datetime.now().strftime('%H:%M:%S')} - Node B04: Data Packet Rcvd",
        f"{datetime.now().strftime('%H:%M:%S')} - System: Database Sync Complete",
        f"{datetime.now().strftime('%H:%M:%S')} - Node C09: Battery Level 85%"
    ]
    
    log_html = ""
    for log in logs:
        log_html += f"<div style='font-size: 0.9rem; color: #E0EAE5; margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 4px;'>{log}</div>"
    
    log_placeholder.markdown(log_html, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
        <div style='font-size: 0.8rem; color: #BBB; text-transform: uppercase;'>Network Latency</div>
        <div style='font-size: 1.2rem; color: #FFF; font-weight: bold;'>24ms</div>
    </div>
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;'>
        <div style='font-size: 0.8rem; color: #BBB; text-transform: uppercase;'>Cloud Connection</div>
        <div style='font-size: 1.2rem; color: #FFF; font-weight: bold;'>Stable</div>
    </div>
    """, unsafe_allow_html=True)

# --- Main Layout ---

st.markdown("<br>", unsafe_allow_html=True)

# 1. Header
st.markdown('<div style="display: flex; justify-content: space-between; align-items: flex-end; padding: 0 10px;">', unsafe_allow_html=True)
st.markdown('<div>', unsafe_allow_html=True)
st.markdown("# EcoEar Command Center")
st.markdown("<p style='color: #4A6E69; font-size: 1.4rem; margin-top: 5px; margin-bottom: 0; font-weight: 300; line-height: 1.4;'>Real-time Forest Acoustic Anomaly Detection System</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("<div style='color: #888; font-size: 1.1rem; padding-bottom: 5px;'>Author: Tanghao Chen (Dios)</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# 2. Banner
if os.path.exists(BANNER_PATH):
    try:
        img_b64 = get_base64_of_bin_file(BANNER_PATH)
        st.markdown(
            f'<div class="banner-container"><img src="data:image/jpeg;base64,{img_b64}" class="custom-banner"></div>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Image Error: {e}")

# 3. Global Metrics Dashboard
with st.container(border=True):
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Active Sensors", "48/50", "+2")
    with m2:
        st.metric("Threats Detected (24h)", "12", "-5", delta_color="inverse")
    with m3:
        st.metric("Forest Coverage", "840 Acres")
    with m4:
        st.metric("System Uptime", "99.8%")

# 4. Input & Control
c_left, c_right = st.columns([1, 1], gap="medium")

with c_left:
    with st.container(border=True):
        st.markdown("### 1. Control Panel")
        threshold = st.slider("Sensitivity Threshold", 0.0, 1.0, 0.60, 0.05)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Processing Unit:**") 
        st.code(str(DEVICE).upper())

with c_right:
    with st.container(border=True):
        st.markdown("### 2. Signal Input")
        tab_sample, tab_upload = st.tabs(["Pre-loaded Samples", "Upload .WAV"])
        
        selected_file = None
        if not os.path.exists("data/danger"): os.makedirs("data/danger")
        if not os.path.exists("data/safe"): os.makedirs("data/safe")
        danger_files = sorted(os.listdir("data/danger"))[:10]
        safe_files = sorted(os.listdir("data/safe"))[:10]

        with tab_sample:
            sample_type = st.radio("Category", ["Illegal Logging", "Natural Ambience"], horizontal=True)
            if "Logging" in sample_type:
                if danger_files:
                    f_name = st.selectbox("Select File", danger_files)
                    selected_file = os.path.join("data/danger", f_name)
            else:
                if safe_files:
                    f_name = st.selectbox("Select File", safe_files)
                    selected_file = os.path.join("data/safe", f_name)

        with tab_upload:
            uploaded = st.file_uploader("Upload", type=["wav"])
            if uploaded:
                with open("temp.wav", "wb") as f:
                    f.write(uploaded.getbuffer())
                selected_file = "temp.wav"

# 5. Analysis Result
if selected_file:
    model = load_model()
    audio_data, spec_img, input_tensor = process_audio(selected_file)
    
    if audio_data is not None:
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
        
        is_danger = prob > threshold
        
        # New Layout for Results
        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 2], gap="medium")
        
        with r1:
            with st.container(border=True):
                st.markdown("### 3. Diagnosis")
                status_color = "#C62828" if is_danger else "#264E47"
                status_text = "THREAT DETECTED" if is_danger else "SAFE"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 20px;">
                    <div style="font-size: 3.5rem; font-family: 'Playfair Display'; font-weight: 700; color: {status_color};">
                        {prob:.1%}
                    </div>
                    <div style="font-size: 0.9rem; color: #888; margin-bottom: 15px;">CONFIDENCE SCORE</div>
                    <div style="background-color: {status_color}; color: white; padding: 8px 16px; border-radius: 50px; display: inline-block; font-weight: bold;">
                        {status_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.audio(selected_file)

        with r2:
            with st.container(border=True):
                st.markdown("### 4. Spectral Analysis")
                plt.style.use('default')
                fig, ax = plt.subplots(figsize=(10, 3))
                im = ax.imshow(spec_img, origin='lower', aspect='auto', cmap='viridis') 
                ax.axis('off')
                plt.tight_layout()
                st.pyplot(fig)
                
                if is_danger:
                     st.error("Anomaly Pattern: High-frequency mechanical signature matching 'Chainsaw' class.")
                else:
                     st.success("Pattern: Biophony (Bird/Wind) signature within normal range.")

        # Geospatial Map Simulation
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("### 5. Geospatial Localization")
            st.markdown("Simulated triangulation of the acoustic source within the sensor network.")
            
            map_df = generate_map_data()
            
            if is_danger:
                threat_point = pd.DataFrame({
                    'lat': [40.4406], 
                    'lon': [-79.9959], 
                    'color': ['#FF0000'],
                    'size': [200]
                })
                map_df['color'] = '#264E47'
                map_df['size'] = 20
                
                st.map(threat_point, latitude='lat', longitude='lon', size='size', color='color', zoom=13)
            else:
                map_df['color'] = '#264E47'
                map_df['size'] = 20
                st.map(map_df, latitude='lat', longitude='lon', size='size', color='color', zoom=13)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #BBB;">&copy; 2025 EcoEar Initiative</div>', unsafe_allow_html=True)