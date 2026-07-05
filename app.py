import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
import os
from datetime import datetime, timedelta

# Set web page configuration
st.set_page_config(page_title="2026 World Cup Predictor", page_icon="⚽", layout="wide")

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

# 1. LOAD DATA
@st.cache_resource
def load_data():
    if os.path.exists("results.csv"):
        df = pd.read_csv("results.csv")
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None

df = load_data()

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Tournament Simulation", "⚔️ Head-to-Head", "📅 Match History & Schedule"])

with tab1:
    st.write("Full Tournament Simulation Logic here...")

with tab2:
    st.write("Matchup Arena Logic here...")

with tab3:
    st.subheader("Match History & Upcoming Schedule")
    
    if df is not None:
        today = pd.to_datetime('2026-07-06')
        
        # Historical Matches
        done_matches = df[df['date'] < today].sort_values('date', ascending=False).head(10)
        st.markdown("### 🏁 Recent Historical Matches")
        
        display_df = done_matches[['date', 'home_team', 'home_score', 'away_score', 'away_team']].copy()
        display_df['Match'] = display_df.apply(lambda r: f"{get_flag(r['home_team'])} {r['home_score']} - {r['away_score']} {get_flag(r['away_team'])}", axis=1)
        st.table(display_df[['date', 'Match']])
    else:
        st.error("results.csv not found!")
    
    st.divider()
    
    # Upcoming Schedule
    st.markdown("### 🗓️ Upcoming Schedule (Predicted)")
    start_date = datetime(2026, 7, 7)
    future_matches = [{"Date": (start_date + timedelta(days=i)).strftime("%b %d"), "Match": "TBD vs TBD"} for i in range(5)]
    st.table(pd.DataFrame(future_matches))
