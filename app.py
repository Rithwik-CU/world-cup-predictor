import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
import os
import sqlite3
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
# 1. UPGRADE 3: PERSISTENT STORAGE (SQLite)
# ==========================================
def init_db():
    """Initialize a local SQLite database to store user-added matches."""
    conn = sqlite3.connect('user_matches.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS matches 
                 (date TEXT, home_team TEXT, away_team TEXT, home_score INTEGER, away_score INTEGER)''')
    conn.commit()
    return conn

conn = init_db()

# ==========================================
# 2. UPGRADED AI ENGINE (Feature Engineering & Evaluation)
# ==========================================
class AdvancedEloSystem:
    def __init__(self):
        self.ratings = {}
        # UPGRADE 1: Feature Engineering (Recent Form Tracking)
        self.recent_form = {} 
        # UPGRADE 2: Model Evaluation Tracking
        self.correct_predictions = 0
        self.total_predictions = 0

    def get_rating(self, team):
        if team == 'USA': team = 'United States'
        return self.ratings.get(team, 1500)
    
    def get_form_bonus(self, team):
        """Calculates an Elo modifier based on the last 5 matches."""
        history = self.recent_form.get(team, [])
        if not history: return 0
        win_rate = sum(history) / len(history)
        return (win_rate - 0.5) * 100  # Gives max +50 for all wins, -50 for all losses

    def get_win_prob(self, team_a, team_b):
        """Calculates win probability including base Elo AND recent form bonus."""
        ra = self.get_rating(team_a) + self.get_form_bonus(team_a)
        rb = self.get_rating(team_b) + self.get_form_bonus(team_b)
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    def update_ratings(self, team_a, team_b, goal_a, goal_b):
        if team_a == 'USA': team_a = 'United States'
        if team_b == 'USA': team_b = 'United States'
        
        # --- Evaluation Phase (Before Updating) ---
        pre_match_prob_a = self.get_win_prob(team_a, team_b)
        actual_a = 1 if goal_a > goal_b else (0 if goal_a < goal_b else 0.5)
        
        # Did the model correctly predict the favorite?
        if (pre_match_prob_a >= 0.5 and actual_a > 0.5) or (pre_match_prob_a < 0.5 and actual_a < 0.5) or (actual_a == 0.5):
            self.correct_predictions += 1
        self.total_predictions += 1

        # --- Elo Update Phase ---
        ra, rb = self.get_rating(team_a), self.get_rating(team_b)
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        g_mult = 1 if abs(goal_a - goal_b) <= 1 else 1.5
        
        self.ratings[team_a] = ra + 40 * g_mult * (actual_a - ea)
        self.ratings[team_b] = rb + 40 * g_mult * ((1 - actual_a) - (1 - ea))
        
        # --- Form Tracking Phase ---
        if team_a not in self.recent_form: self.recent_form[team_a] = []
        if team_b not in self.recent_form: self.recent_form[team_b] = []
        
        self.recent_form[team_a].append(actual_a)
        self.recent_form[team_b].append(1 - actual_a)
        self.recent_form[team_a] = self.recent_form[team_a][-5:] # Keep only last 5
        self.recent_form[team_b] = self.recent_form[team_b][-5:]

@st.cache_data
def load_and_train_elo():
    elo = AdvancedEloSystem()
    df_main, df_db = None, None
    
    # 1. Load Original CSV
    if os.path.exists("results.csv"):
        df_main = pd.read_csv("results.csv", low_memory=False).dropna(subset=['home_score', 'away_score'])
    
    # 2. Load Persisted Database Matches
    df_db = pd.read_sql_query("SELECT * FROM matches", conn)
    
    # Combine data
    if df_main is not None:
        df_combined = pd.concat([df_main, df_db], ignore_index=True)
        # Train model
        for row in df_combined.itertuples(index=False):
            elo.update_ratings(row.home_team, row.away_team, int(row.home_score), int(row.away_score))
        return elo, df_combined
    return elo, df_db

model, full_df = load_and_train_elo()

# Field setup
wc_groups = {'Group A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'], 'Group B': ['Canada', 'Switzerland', 'Qatar', 'Bosnia and Herzegovina'], 'Group C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'], 'Group D': ['United States', 'Paraguay', 'Australia', 'Turkey'], 'Group E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'], 'Group F': ['Netherlands', 'Japan', 'Tunisia', 'Sweden'], 'Group G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'], 'Group H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'], 'Group I': ['France', 'Senegal', 'Norway', 'Iraq'], 'Group J': ['Argentina', 'Algeria', 'Austria', 'Jordan'], 'Group K': ['Portugal', 'Uzbekistan', 'Colombia', 'DR Congo'], 'Group L': ['England', 'Croatia', 'Ghana', 'Panama']}
all_teams = sorted([team for group in wc_groups.values() for team in group])
field_elos = {team: model.get_rating(team) for team in all_teams}

# ==========================================
# 3. INTERACTIVE DASHBOARD TABS
# ==========================================
st.title("🏆 Advanced World Cup ML Predictor")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tournament Sim", "⚔️ Head-to-Head", "📈 Model Evaluation", "💾 Database Input"
])

# --- TAB 1: TOURNAMENT SIMULATION ---
with tab1:
    st.subheader("Monte Carlo Simulation Engine")
    iterations = st.slider("Simulations", 500, 5000, 1500)
    if st.button("🚀 Run Simulation"):
        with st.spinner("Running stochastic probability paths..."):
            counts = {team: 0 for team in all_teams}
            for _ in range(iterations):
                # Weighted probability using active Elo ratings
                winner = random.choices(all_teams, weights=[field_elos[t] for t in all_teams])[0]
                counts[winner] += 1
                
            results_df = pd.DataFrame([
                {"Team": get_flag(team), "Win Prob (%)": (wins / iterations) * 100} 
                for team, wins in counts.items()
            ]).sort_values(by="Win Prob (%)", ascending=False)
            
            fig = px.bar(results_df.head(15), x="Win Prob (%)", y="Team", orientation='h', title="Top 15 Favorites")
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: HEAD TO HEAD ---
with tab2:
    st.subheader("Simulate Knockout Match")
    col1, col2 = st.columns(2)
    t1 = col1.selectbox("Team A", all_teams, format_func=get_flag, index=all_teams.index("United States"))
    t2 = col2.selectbox("Team B", all_teams, format_func=get_flag, index=all_teams.index("England"))
    
    if st.button("Simulate Match"):
        prob_a = model.get_win_prob(t1, t2)
        prob_b = 1 - prob_a
        winner = t1 if random.random() < prob_a else t2
        
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{get_flag(t1)} Win", f"{prob_a*100:.1f}%")
        c2.metric("Result", f"🏆 {get_flag(winner)}")
        c3.metric(f"{get_flag(t2)} Win", f"{prob_b*100:.1f}%")

# --- TAB 3: MODEL EVALUATION ---
with tab3:
    st.subheader("Model Performance & Diagnostics")
    st.write("This tab evaluates the Elo algorithm's historical accuracy using Backtesting.")
    
    accuracy = (model.correct_predictions / max(model.total_predictions, 1)) * 100
    
    c1, c2 = st.columns(2)
    c1.metric("Historical Accuracy", f"{accuracy:.1f}%")
    c2.metric("Matches Processed", f"{model.total_predictions:,}")
    
    st.divider()
    st.markdown("#### Global Power Rankings (Top 10)")
    rankings = pd.DataFrame([
        {"Team": get_flag(t), "Base Elo": int(model.get_rating(t)), "Form Bonus": int(model.get_form_bonus(t))}
        for t in all_teams
    ]).sort_values(by="Base Elo", ascending=False).head(10).reset_index(drop=True)
    rankings.index += 1
    st.table(rankings)

# --- TAB 4: PERSISTENT DATABASE ---
with tab4:
    st.subheader("Log New Match Results")
    st.write("Add a new match result below. It will be saved permanently to the SQLite database and instantly retrain the AI!")
    
    with st.form("add_match"):
        date_input = st.date_input("Match Date")
        colA, colB = st.columns(2)
        team_home = colA.selectbox("Home Team", all_teams, key='h')
        score_home = colA.number_input("Home Score", 0, 15, 0)
        team_away = colB.selectbox("Away Team", all_teams, key='a')
        score_away = colB.number_input("Away Score", 0, 15, 0)
        
        submitted = st.form_submit_button("💾 Save Match & Retrain AI")
        if submitted:
            c = conn.cursor()
            c.execute("INSERT INTO matches VALUES (?, ?, ?, ?, ?)", 
                      (date_input.strftime("%Y-%m-%d"), team_home, team_away, score_home, score_away))
            conn.commit()
            st.cache_data.clear() # Forces AI to retrain
            st.success(f"Match Logged! {team_home} {score_home} - {score_away} {team_away}. Model Retrained.")
            st.rerun()
