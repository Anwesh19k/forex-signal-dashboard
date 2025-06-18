import streamlit as st
import pandas as pd
from datetime import datetime

# === IMPORTS ===
from one_hour import run_signal_engine as run_one_hour
from one_hour_pro import run_signal_engine as run_one_hour_pro
from one_hour_pro_plus import run_signal_engine as run_one_hour_pro_plus  # Make sure this exists

# === CONFIG ===
st.set_page_config(
    page_title="Forex Signal Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === THEME CSS ===
def set_custom_theme(mode):
    if mode == "Dark":
        st.markdown("""
            <style>
                body {
                    background-color: #0e1117;
                    color: #FFFFFF;
                }
                .stDataFrame thead tr th {
                    color: #FFFFFF;
                    background-color: #1c1c1c;
                }
                .stDataFrame tbody tr td {
                    background-color: #1c1c1c;
                    color: #FFFFFF;
                }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
                .stDataFrame tbody tr td {
                    background-color: #FAFAFA;
                    color: #000000;
                }
            </style>
        """, unsafe_allow_html=True)

# === SIDEBAR ===
mode = st.sidebar.radio("🌗 Theme Mode", ["Light", "Dark"], index=0)
set_custom_theme(mode)

# === TITLE ===
st.title("📊 Forex Signal Dashboard (1H, Pro & Pro+)")
st.markdown("Get real-time signals from three AI models: **Standard**, **Pro**, and **Pro+**.")
st.caption("✅ Fully optimized for Desktop and Mobile screens.")

# === REFRESH ===
if st.button("🔄 Refresh Dashboards"):
    st.session_state['last_refreshed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.success("🔁 Dashboard refreshed!")

st.markdown(f"🕒 **Last Refreshed:** `{st.session_state.get('last_refreshed', 'Not yet refreshed')}`")

# === TABS (Mobile/Desktop Friendly) ===
tab1, tab2, tab3 = st.tabs(["📘 1 Hour Model", "📗 1 Hour Pro", "📙 1 Hour Pro+"])

with tab1:
    df1 = run_one_hour()
    st.dataframe(df1, use_container_width=True)

with tab2:
    df2 = run_one_hour_pro()
    st.dataframe(df2, use_container_width=True)

with tab3:
    df3 = run_one_hour_pro_plus()
    st.dataframe(df3, use_container_width=True)

        


