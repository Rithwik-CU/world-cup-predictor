import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
import os
from datetime import datetime, timedelta

# Set web page configuration
st.set_page_config(page_title="2026 World Cup Predictor", page_icon="🏆", layout="wide")

# ==========================================
# 0. UI HELPERS & EMOJI DICTIONARY
# ==========================================
FLAGS = {
    'Mexico': '🇲🇽', 'South Africa': '🇿🇦', 'South Korea': '🇰🇷', 'Czech Republic': '🇨🇿',
    'Canada': '🇨🇦', 'Switzerland': '🇨🇭', 'Qatar': '🇶🇦', 'Bosnia and Herzegovina': '🇧🇦',
    'Brazil': '🇧🇷', 'Morocco': '🇲🇦', 'Haiti': '🇭🇹', 'Scotland': '🏴󠁧󠁢󠁳󠁣󠁴󠁿',
    'United States': '🇺🇸', 'Paraguay': '🇵🇾', 'Australia': '🇦🇺', 'Turkey': '🇹🇷',
    'Germany': '🇩🇪', 'Curaçao': '🇨🇼', 'Ivory Coast': '🇨🇮', 'Ecuador': '🇪🇨',
    'Netherlands': '🇳🇱', 'Japan': '🇯🇵', 'Tunisia': '🇹🇳', 'Sweden': '🇸🇪',
    'Belgium': '🇧🇪', 'Egypt': '🇪🇬', 'Iran': '🇮🇷', 'New Zealand': '🇳🇿',
    'Spain': '🇪🇸', 'Cape Verde': '🇨🇻', 'Saudi Arabia': '🇸🇦', 'Uruguay': '🇺🇾',
    'France': '🇫🇷', 'Senegal': '🇸🇳', 'Norway': '🇳🇴', 'Iraq': '🇮🇶',
    'Argentina': '🇦🇷', 'Algeria': '🇩🇿', 'Austria': '🇦🇹', 'Jordan': '🇯🇴',
    'Portugal': '🇵🇹', 'Uzbekistan': '🇺🇿', 'Colombia': '🇨🇴', 'DR Congo': '🇨🇩',
    'England': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'Croatia': '🇭🇷', 'Ghana': '🇬🇭', 'Panama': '🇵🇦'
}

def get_flag(team_name):
    return f"{FLAGS.get(team_name, '🏳️')} {team_name}"

# ==========================================
# 1. SETUP
# ==========================================
st.title("🏆 Interactive 2026 World Cup Predictor")
all_teams = sorted(list(FLAGS.keys()))

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Full Tournament Simulation", "⚔️ Head-to-Head Matchup Arena", "📅 Tournament Schedule"])

with tab1:
    st.write("Full simulation logic here...")

with tab2:
    st.subheader("Simulate a Single Knockout Match")
    col1, col2 = st.columns(2)
    # Using format_func to show flags in the dropdowns
    t1 = col1.selectbox("Team A", all_teams, format_func=get_flag)
    t2 = col2.selectbox("Team B", all_teams, format_func=get_flag)
    if st.button("Simulate"):
        st.write(f"Simulating {get_flag(t1)} vs {get_flag(t2)}")

with tab3:
    st.subheader("Tournament Schedule")
    start_date = datetime(2026, 6, 11)
    match_list = []
    # Example schedule entry using flags
    match_list.append({"Date": start_date.strftime("%b %d"), "Match": f"{get_flag('United States')} vs {get_flag('Mexico')}", "Status": "Opening"})
    st.table(pd.DataFrame(match_list))
