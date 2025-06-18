
import streamlit as st
import pandas as pd
from datetime import datetime
import time

from one_hour import run_signal_engine as run_one_hour
from one_hour_pro import run_signal_engine as run_one_hour_pro

# Page config
st.set_page_config(page_title="Forex Signal Dashboard", layout="wide", initial_sidebar_state="expanded")

# CSS for light and dark themes
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

# Sidebar mode selection
mode = st.sidebar.radio("🌗 Theme Mode", ["Light", "Dark"], index=0)
set_custom_theme(mode)

# Title and Info
st.title("📊 Forex Signal Dashboard (1H & 1H Pro)")
st.markdown("Get real-time signals from two AI models: **Standard** and **Pro**.")
st.caption("✅ Built to work seamlessly across PC and Mobile devices.")

# Refresh button
if st.button("🔄 Refresh Dashboards"):
    st.session_state['last_refreshed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.success("🔁 Dashboard refreshed!")

st.markdown(f"🕒 **Last Refreshed:** `{st.session_state.get('last_refreshed', 'Not yet refreshed')}`")

# Responsive layout: switch to one-column if screen is narrow
if st.columns(2)[0].beta_container()._parent.width < 768:
    # Mobile View: Single Column
    st.subheader("📘 1 Hour Model")
    df1 = run_one_hour()
    st.dataframe(df1, use_container_width=True)

    st.subheader("📗 1 Hour Pro Model")
    df2 = run_one_hour_pro()
    st.dataframe(df2, use_container_width=True)
   
    st.subheader("📗 1 Hour Pro Model+")
    df2 = run_one_hour_pro_plus()
    st.dataframe(df2, use_container_width=True)
else:
    # Desktop View: Two Columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📘 1 Hour Model")
        df1 = run_one_hour()
        st.dataframe(df1, use_container_width=True)

    with col2:
        st.subheader("📗 1 Hour Pro Model")
        df2 = run_one_hour_pro()
        st.dataframe(df2, use_container_width=True)
    with col2:
        st.subheader("📗 1 Hour Pro Model+")
        df2 = run_one_hour_pro_plus()
        st.dataframe(df2, use_container_width=True)

        


