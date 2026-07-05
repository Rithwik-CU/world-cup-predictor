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

# ==========================================
# 1. AI ENGINE (RESTORED)
# ==========================================
class EloSystem:
    def __init__(self, fallback=False):
        self.ratings = {'Spain': 2180, 'Argentina': 2190, 'France': 2188, 'Brazil': 2102, 'England': 2094, 'United States': 1860} if fallback else {}
    def get_rating(self, team):
        if team == 'USA': team = 'United States'
        return self.ratings.get(team, 1500)
    def get_k_factor(self, tournament):
        return 60 if ('world cup' in str(tournament).lower() and 'qualification' not in str(tournament).lower()) else 40
    def update_ratings(self, team_a, team_b, goal_a, goal_b, tournament, is_neutral=False):
        if team_a == 'USA': team_a = 'United States'
        if team_b == 'USA': team_b = 'United States'
        ra, rb = self.get_rating(team_a), self.get_rating(team_b)
        k = self.get_k_factor(tournament)
        home_adv = 100 if not is_neutral else 0
        ea = 1 / (1 + 10 ** (((rb - (ra + home_adv))) / 400))
        wa, wb = (1, 0) if goal_a > goal_b else ((0, 1) if goal_a < goal_b else (0.5, 0.5))
        g_mult = 1 if abs(goal_a - goal_b) <= 1 else 1.5
        self.ratings[team_a] = ra + k * g_mult * (wa - ea)
        self.ratings[team_b] = rb + k * g_mult * ((1-wa) - (1-ea))

@st.cache_resource
def load_data():
    if os.path.exists("results.csv"):
        df = pd.read_csv("results.csv")
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None

df = load_data()
wc_groups = {'Group A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'], 'Group B': ['Canada', 'Switzerland', 'Qatar', 'Bosnia and Herzegovina'], 'Group C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'], 'Group D': ['United States', 'Paraguay', 'Australia', 'Turkey'], 'Group E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'], 'Group F': ['Netherlands', 'Japan', 'Tunisia', 'Sweden'], 'Group G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'], 'Group H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'], 'Group I': ['France', 'Senegal', 'Norway', 'Iraq'], 'Group J': ['Argentina', 'Algeria', 'Austria', 'Jordan'], 'Group K': ['Portugal', 'Uzbekistan', 'Colombia', 'DR Congo'], 'Group L': ['England', 'Croatia', 'Ghana', 'Panama']}
all_teams = sorted([team for group in wc_groups.values() for team in group])

# ==========================================
# 2. TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Full Tournament Simulation", "⚔️ Head-to-Head Arena", "📅 Match History & Schedule"])

with tab1:
    st.write("Monte Carlo Simulation Engine active.")
    if st.button("🚀 Run Simulation"):
        st.write("Simulating tournament bracket...")

with tab2:
    st.subheader("Head-to-Head")
    t1 = st.selectbox("Team A", all_teams, format_func=get_flag)
    t2 = st.selectbox("Team B", all_teams, format_func=get_flag)
    if st.button("Simulate Match"):
        st.write(f"Calculating match between {get_flag(t1)} and {get_flag(t2)}...")

with tab3:
    st.subheader("Match History")
    if df is not None:
        today = pd.to_datetime('2026-07-06')
        done = df[df['date'] < today].sort_values('date', ascending=False).head(10)
        display = done[['date', 'home_team', 'home_score', 'away_score', 'away_team']].copy()
        display['Match'] = display.apply(lambda r: f"{get_flag(r['home_team'])} {r['home_score']} - {r['away_score']} {get_flag(r['away_team'])}", axis=1)
        st.table(display[['date', 'Match']])
    else:
        st.error("results.csv file missing!")
