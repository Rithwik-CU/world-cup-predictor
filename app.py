import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
import os

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
    'Spain': '🇪🇸', 'Cape Verde': '🇨🇻', 'Saudi Arabia': '🇸🇦', 'Uruguay': '🇾🇺',
    'France': '🇫🇷', 'Senegal': '🇸🇳', 'Norway': '🇳🇴', 'Iraq': '🇮🇶',
    'Argentina': '🇦🇷', 'Algeria': '🇩🇿', 'Austria': '🇦🇹', 'Jordan': '🇯🇴',
    'Portugal': '🇵🇹', 'Uzbekistan': '🇺🇿', 'Colombia': '🇨🇴', 'DR Congo': '🇨🇩',
    'England': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'Croatia': '🇭🇷', 'Ghana': '🇬🇭', 'Panama': '🇵🇦'
}

def get_flag(team_name):
    return f"{FLAGS.get(team_name, '🏳️')} {team_name}"

# ==========================================
# 1. AI ENGINE
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
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        wa, wb = (1, 0) if goal_a > goal_b else ((0, 1) if goal_a < goal_b else (0.5, 0.5))
        g_mult = 1 if abs(goal_a - goal_b) <= 1 else 1.5
        self.ratings[team_a] = ra + k * g_mult * (wa - ea)
        self.ratings[team_b] = rb + k * g_mult * ((1-wa) - (1-ea))

@st.cache_resource
def load_and_train_elo():
    csv_filename = "results.csv"
    if not os.path.exists(csv_filename): return EloSystem(fallback=True), None
    df = pd.read_csv(csv_filename, low_memory=False).dropna(subset=['home_score', 'away_score', 'home_team', 'away_team'])
    elo = EloSystem(fallback=False)
    for row in df.itertuples(index=False):
        elo.update_ratings(row.home_team, row.away_team, int(row.home_score), int(row.away_score), getattr(row, 'tournament', 'Friendly'))
    return elo, df

elo_model, _ = load_and_train_elo()
wc_groups = {'Group A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'], 'Group B': ['Canada', 'Switzerland', 'Qatar', 'Bosnia and Herzegovina'], 'Group C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'], 'Group D': ['United States', 'Paraguay', 'Australia', 'Turkey'], 'Group E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'], 'Group F': ['Netherlands', 'Japan', 'Tunisia', 'Sweden'], 'Group G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'], 'Group H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'], 'Group I': ['France', 'Senegal', 'Norway', 'Iraq'], 'Group J': ['Argentina', 'Algeria', 'Austria', 'Jordan'], 'Group K': ['Portugal', 'Uzbekistan', 'Colombia', 'DR Congo'], 'Group L': ['England', 'Croatia', 'Ghana', 'Panama']}
all_teams = sorted([team for group in wc_groups.values() for team in group])
field_elos = {team: elo_model.get_rating(team) for team in all_teams}

# ==========================================
# 2. TABS
# ==========================================
tab1, tab2 = st.tabs(["📊 Full Tournament Simulation", "⚔️ Head-to-Head Matchup Arena"])

with tab1:
    st.subheader("Monte Carlo Simulation Engine")
    iterations = st.slider("Number of Simulations", 500, 5000, 1500)
    if st.button("🚀 Run Full Tournament Simulation"):
        with st.spinner("Simulating..."):
            counts = {team: 0 for team in all_teams}
            for _ in range(iterations):
                winner = random.choices(all_teams, weights=[field_elos[t] for t in all_teams])[0]
                counts[winner] += 1
            results_df = pd.DataFrame([{"Team": get_flag(team), "Win Probability (%)": (wins / iterations) * 100} for team, wins in counts.items()]).sort_values(by="Win Probability (%)", ascending=False)
            fig = px.bar(results_df.head(15), x="Win Probability (%)", y="Team", orientation='h', title="Top 15 Favorites")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Simulate a Single Knockout Match")
    col1, col2 = st.columns(2)
    t1 = col1.selectbox("Team A", all_teams, format_func=get_flag)
    t2 = col2.selectbox("Team B", all_teams, format_func=get_flag)
    
    if st.button("Simulate Match"):
        # Calculate Probabilities
        elo_a = field_elos[t1]
        elo_b = field_elos[t2]
        
        # Expected scores based on Elo difference
        prob_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        prob_b = 1 - prob_a
        
        # Display Results
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{get_flag(t1)} Win", f"{prob_a*100:.1f}%")
        c2.metric("Draw", "—")
        c3.metric(f"{get_flag(t2)} Win", f"{prob_b*100:.1f}%")
        
        # Simple Outcome
        winner = t1 if random.random() < prob_a else t2
        st.write(f"### Predicted Winner: {get_flag(winner)}")
