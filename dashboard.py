import streamlit as st
import numpy as np
import pandas as pd
import time
import subprocess
import re
from stable_baselines3 import PPO
from src.env import PacerEnv

# --- CONFIGURATION ---
st.set_page_config(page_title="RL vs TCP: Congestion Control Dashboard", layout="wide")

# Modern Styling
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 10px 15px;
        border-radius: 10px;
        border: 1px solid #334155;
        margin-bottom: 5px;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
    }
    .stMetric [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        color: #94a3b8;
    }
    .winner-text {
        color: #10b981 !important; /* Emerald Green */
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 PacerRL: AI-Driven Congestion Control")
st.markdown("Comparing **PacerRL (Our Agent)** with **Standard TCP (AIMD)** across three core dimensions.")

# --- SIDEBAR & SETTINGS ---
with st.sidebar:
    st.header("⚙️ System Settings")
    target_ip = st.text_input("Target IP Address", value="127.0.0.1")
    test_mode = st.radio("Mode", ["Simulation (Demo)", "Real Network (iperf3)"])
    st.markdown("---")
    st.subheader("🏎️ Hardware Throttling")
    bitrate_limit = st.slider("Link Capacity (Mbps)", 10, 1000, 500)
    
    if test_mode == "Simulation (Demo)":
        st.caption("🧪 **Simulation**: Uses mathematical models to predict network behavior. Ideal for fast demos and training.")
        congestion_level = st.slider("Network Noise (%)", 0, 100, 25)
    else:
        st.caption("🌐 **Real Network**: Uses `iperf3` and `ping` to measure actual traffic on your hardware. Slower but realistic.")
        st.info(f"Throttling Real Network to **{bitrate_limit} Mbps** via iperf3 -b.")

    st.markdown("---")
    st.subheader("📊 Test Parameters")
    total_steps = st.slider("Test Duration (Steps)", 10, 200, 50)
    step_delay = st.slider("Step Refresh Rate (s)", 0.01, 1.0, 0.1)

# --- STATE MANAGEMENT ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- METRIC HELPERS ---
# Using shared PacerEnv for consistency
# (fetch_metrics logic is now inside PacerEnv)

# --- DASHBOARD UI COMPONENTS ---
col_stats1, col_stats2 = st.columns(2)

with col_stats1:
    st.subheader("🤖 PacerRL Agent")
    rl_r1c1, rl_r1c2 = st.columns(2)
    rl_tr_ui = rl_r1c1.empty()
    rl_de_ui = rl_r1c2.empty()
    rl_r2c1, rl_r2c2 = st.columns(2)
    rl_lo_ui = rl_r2c1.empty()
    rl_eff_ui = rl_r2c2.empty()

with col_stats2:
    st.subheader("🌐 Standard TCP")
    tcp_r1c1, tcp_r1c2 = st.columns(2)
    tcp_tr_ui = tcp_r1c1.empty()
    tcp_de_ui = tcp_r1c2.empty()
    tcp_r2c1, tcp_r2c2 = st.columns(2)
    tcp_lo_ui = tcp_r2c1.empty()
    tcp_eff_ui = tcp_r2c2.empty()

st.markdown("---")
st.subheader("📉 Live Comparison Charts")
chart_col1, chart_col2, chart_col3 = st.columns(3)
throughput_chart = chart_col1.empty()
delay_chart = chart_col2.empty()
loss_chart = chart_col3.empty()
window_chart = st.empty()

# --- MAIN EXECUTION ---
if st.button("🔥 Start Live Comparison"):
    st.session_state.history = []
    
    # Load RL Model
    try:
        model = PPO.load("cc_ppo_optimized")
        has_model = True
    except:
        st.error("Optimized model not found! Please run the training notebook first.")
        has_model = False
    
    if has_model:
        # Initialize Environments
        mode_flag = "sim" if test_mode == "Simulation (Demo)" else "real"
        cong = congestion_level if mode_flag == "sim" else 0
        
        # RL Env
        rl_env = PacerEnv(target_ip=target_ip, mode=mode_flag, congestion_level=cong, max_bitrate=f"{bitrate_limit}M")
        rl_obs, _ = rl_env.reset()
        
        # TCP Env (separate instance to avoid state mixing)
        tcp_env = PacerEnv(target_ip=target_ip, mode=mode_flag, congestion_level=cong, max_bitrate=f"{bitrate_limit}M")
        tcp_env.reset()
        tcp_window = 256
        
        # Progress Bar
        progress = st.progress(0)
        
        for s in range(total_steps):
            # 1. PacerRL Step
            action, _ = model.predict(rl_obs, deterministic=True)
            rl_obs, _, _, _, info = rl_env.step(action)
            rl_tr, rl_de, rl_lo = info["tr"], info["de"], info["lo"]
            rl_win = rl_env.window_sizes[rl_env.idx]
            
            # Small cooldown for the iperf3 server if in 'real' mode
            if mode_flag == "real":
                time.sleep(0.1)
            
            # 2. TCP Step (Simplified AIMD logic)
            tcp_tr, tcp_de, tcp_lo = tcp_env.fetch_metrics(tcp_window)
            
            if tcp_de > 120 or tcp_lo > 0:
                tcp_window = max(64, tcp_window // 2)
            else:
                tcp_window = min(4096, tcp_window + 128)
            
            # 3. Update History
            rl_eff = rl_tr / (rl_de / 10.0) if rl_de > 0 else 0
            tcp_eff = tcp_tr / (tcp_de / 10.0) if tcp_de > 0 else 0
            
            st.session_state.history.append({
                "Step": s,
                "RL Throughput": rl_tr, "TCP Throughput": tcp_tr,
                "RL Delay": rl_de, "TCP Delay": tcp_de,
                "RL Loss": rl_lo, "TCP Loss": tcp_lo,
                "RL Efficiency": rl_eff, "TCP Efficiency": tcp_eff,
                "RL Window": rl_win, "TCP Window": tcp_window
            })
            
            # 4. Update UI
            df = pd.DataFrame(st.session_state.history).set_index("Step")
            
            # --- UI UPDATE WITH WINNER HIGHLIGHTING ---
            def get_metric_html(label, value, is_winner, unit=""):
                win_class = "winner-text" if is_winner else ""
                win_icon = " 🏆" if is_winner else ""
                return f"""
                <div class="stMetric">
                    <div data-testid="stMetricLabel">{label}{win_icon}</div>
                    <div data-testid="stMetricValue" class="{win_class}">{value}{unit}</div>
                </div>
                """

            # Update Metrics
            rl_tr_ui.markdown(get_metric_html("Throughput", f"{rl_tr:.1f}", rl_tr > tcp_tr, " M"), unsafe_allow_html=True)
            rl_de_ui.markdown(get_metric_html("Delay", f"{rl_de:.1f}", rl_de < tcp_de, " ms"), unsafe_allow_html=True)
            rl_lo_ui.markdown(get_metric_html("Loss", f"{rl_lo}", rl_lo <= tcp_lo), unsafe_allow_html=True)
            rl_eff_ui.markdown(get_metric_html("Efficiency", f"{rl_eff:.1f}", rl_eff > tcp_eff), unsafe_allow_html=True)
            
            tcp_tr_ui.markdown(get_metric_html("Throughput", f"{tcp_tr:.1f}", tcp_tr > rl_tr, " M"), unsafe_allow_html=True)
            tcp_de_ui.markdown(get_metric_html("Delay", f"{tcp_de:.1f}", tcp_de < rl_de, " ms"), unsafe_allow_html=True)
            tcp_lo_ui.markdown(get_metric_html("Loss", f"{tcp_lo}", tcp_lo < rl_lo), unsafe_allow_html=True)
            tcp_eff_ui.markdown(get_metric_html("Efficiency", f"{tcp_eff:.1f}", tcp_eff > rl_eff), unsafe_allow_html=True)
            
            # Charts
            throughput_chart.line_chart(df[["RL Throughput", "TCP Throughput"]])
            delay_chart.line_chart(df[["RL Delay", "TCP Delay"]])
            loss_chart.line_chart(df[["RL Loss", "TCP Loss"]])
            window_chart.line_chart(df[["RL Window", "TCP Window"]])
            
            progress.progress((s + 1) / total_steps)
            time.sleep(step_delay)

st.markdown("---")
st.caption("Developed with ❤️ for Advanced Congestion Control Research")
