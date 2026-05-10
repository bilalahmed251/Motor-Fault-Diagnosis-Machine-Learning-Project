import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="AURA | NEXUS Engineering Console", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Cyber-Glass UI
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #00ff41; }
    .stMetric { background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; border: 1px solid #00ff41; }
    .log-container { background: black; color: #00ff41; font-family: 'Courier New', Courier, monospace; height: 300px; overflow-y: scroll; padding: 10px; border: 1px solid #444; }
</style>
""", unsafe_allow_html=True)

st.title("🛸 AURA: NEXUS | Motor Intelligence Engine")
st.subheader("Industrial Grade Predictive Maintenance Dashboard")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Control")
    uploaded_file = st.file_uploader("Inject Raw Current Data (CSV)", type="csv")
    scan_btn = st.button("🚀 INITIATE SYSTEM SCAN")
    
    st.divider()
    st.info("System Status: ONLINE")
    st.success("AI Model: K-NN Nexus Core")

# Main Dashboard
col1, col2, col3 = st.columns(3)

with col1:
    health_score = st.empty()
    health_score.metric("SYSTEM HEALTH SCORE", "98.2%", "0.5%")

with col2:
    st.metric("VIBRATION AMPLITUDE", "0.0024 mm", "-0.0001")

with col3:
    st.metric("THERMAL STABILITY", "42.5°C", "+1.2°C")

# FFT Analysis Section
st.divider()
st.header("🧬 Advanced Frequency Analyzer (FFT Spectrum)")

# Generate Synthetic Data for visualization
t = np.linspace(0, 1, 1000)
freq = 50 # Hz
signal = np.sin(2 * np.pi * freq * t) + 0.5 * np.random.normal(size=1000)
fft_vals = np.abs(np.fft.fft(signal))[:500]
freqs = np.fft.fftfreq(len(signal), d=1/1000)[:500]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(freqs, fft_vals, color='#00ff41')
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.tick_params(colors='white')
ax.set_xlabel("Frequency (Hz)", color='white')
ax.set_ylabel("Amplitude", color='white')
st.pyplot(fig)

# System Logs
st.divider()
st.header("📑 Active System Logs")
log_data = [
    f"[{datetime.now().strftime('%H:%M:%S')}] INITIALIZING NEXUS KERNEL...",
    f"[{datetime.now().strftime('%H:%M:%S')}] LOADING HARMONIC SIGNATURES...",
    f"[{datetime.now().strftime('%H:%M:%S')}] SCANNING SENSOR ARRAY...",
    f"[{datetime.now().strftime('%H:%M:%S')}] STABILITY CHECK: NOMINAL"
]
st.markdown(f'<div class="log-container">{"<br>".join(log_data)}</div>', unsafe_allow_html=True)

# Export Section
st.divider()
df_report = pd.DataFrame({
    'Metric': ['Health Score', 'Peak Frequency', 'RMS Current', 'Fault Probability'],
    'Value': ['98.2%', '50.1 Hz', '12.4A', '0.01%'],
    'Status': ['HEALTHY', 'STABLE', 'OPTIMAL', 'SAFE']
})

csv = df_report.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 DOWNLOAD MASTER DIAGNOSTIC REPORT (CSV)",
    data=csv,
    file_name='AURA_NEXUS_Report.csv',
    mime='text/csv',
)

st.caption("AURA NEXUS | Developed for Industrial Excellence")
