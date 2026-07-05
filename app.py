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
    'Spain': '🇪🇸', 'Cape Verde': '🇨🇻', 'Saudi Arabia': '🇸🇦', 'Uruguay': '🇺🇾',
    'France': '🇫🇷', 'Senegal': '🇸🇳', 'Norway': '🇳🇴', 'Iraq': '🇮🇶',
    'Argentina': '🇦🇷', 'Algeria': '🇩🇿', 'Austria': '🇦🇹', 'Jordan': '🇯🇴',
    'Portugal': '🇵🇹', 'Uzbekistan': '🇺🇿', 'Colombia': '🇨🇴', 'DR Congo': '🇨🇩',
    'England': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'Croatia': '🇭🇷', 'Ghana': '🇬🇭', 'Panama': '🇵🇦'
}

def get_flag(team_name):
    """Helper function to add a flag emoji to a team name."""
    return f"{FLAGS.get(team_name, '🏳️')} {team_name}"

# 1. GUARANTEED INSTANT UI RENDER
st.title("🏆 Interactive 2026 World Cup Match & Tournament Predictor")
st.markdown("Use the controls in the sidebar to configure the Monte Carlo simulation engine and explore matchups live!")

# ==========================================
# 2. SIDEBAR CONTROLS 
# ==========================================
st.sidebar.header("⚙️ Simulation Settings")
iterations = st.sidebar.slider("Number of Tournament Simulations", min_value=500, max_value=5000, value=1500, step=500)
host_advantage = st.sidebar.toggle("Enable Host Advantage (+100 Elo)", value=True)
baseline_goals = st.sidebar.number_input("Baseline Average Goals per Team", min_value=1.0, max_value=2.5, value=1.35, step=0.05)

# ==========================================
# 3. DEFENSIVE AI ENGINE 
# ==========================================
class EloSystem:
    def __init__(self, fallback=False):
        self.ratings = {
            'Spain': 2180, 'Argentina': 2190, 'France': 2188, 'Brazil': 2102, 
            'England': 2094, 'United States': 1860
        } if fallback else {}

    def get_rating(self, team):
        if team == 'USA': team = 'United States'
        return self.ratings.get(team, 1500)

    def get_k_factor(self, tournament):
        t_str = str(tournament).lower()
        return 60 if ('world cup' in t_str and 'qualification' not in t_str) else (40 if 'qualification' in t_str else 20)

    def update_ratings(self, team_a, team_b, goal_a, goal_b, tournament, is_neutral=False):
        if team_a == 'USA': team_a = 'United States'
        if team_b == 'USA': team_b = 'United States'
        ra, rb = self.get_rating(team_a), self.get_rating(team_b)
        k = self.get_k_factor(tournament)
        home_adv = 100 if not is_neutral else 0
        ea = 1 / (1 + 10 ** (((rb - (ra + home_adv))) / 400))
        eb = 1 - ea
        wa, wb = (1, 0) if goal_a > goal_b else ((0, 1) if goal_a < goal_b else (0.5, 0.5))
        margin = abs(goal_a - goal_b)
        g_mult = 1 if margin <= 1 else (1.5 if margin == 2 else (11 + margin) / 8)
        self.ratings[team_a] = ra + k * g_mult * (wa - ea)
        self.ratings[team_b] = rb + k * g_mult * (wb - eb)

@st.cache_resource
def load_and_train_elo():
    csv_filename = "results.csv"
    if not os.path.exists(csv_filename):
        return EloSystem(fallback=True), "⚠️ 'results.csv' not found. Running in Safe-Mode!"
    try:
        df = pd.read_csv(csv_filename, low_memory=False).dropna(subset=['home_score', 'away_score', 'home_team', 'away_team'])
        elo = EloSystem(fallback=False)
        for row in df.itertuples(index=False):
            elo.update_ratings(row.home_team, row.away_team, int(row.home_score), int(row.away_score), getattr(row, 'tournament', 'Friendly'), is_neutral=row.neutral)
        return elo, f"✅ Successfully trained AI on {len(df):,} historical matches from results.csv!"
    except Exception as e:
        return EloSystem(fallback=True), f"⚠️ Error reading CSV. Switched to Safe-Mode!"

elo_model, status_message = load_and_train_elo()
if "⚠️" in status_message:
    st.warning(status_message)
else:
    st.success(status_message)

# 48-Team Field Definition
wc_groups = {
    'Group A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'Group B': ['Canada', 'Switzerland', 'Qatar', 'Bosnia and Herzegovina'],
    'Group C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'Group D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'Group E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'Group F': ['Netherlands', 'Japan', 'Tunisia', 'Sweden'],
    'Group G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'Group H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'Group I': ['France', 'Senegal', 'Norway', 'Iraq'],
    'Group J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'Group K': ['Portugal', 'Uzbekistan', 'Colombia', 'DR Congo'],
    'Group L': ['England', 'Croatia', 'Ghana', 'Panama']
}
all_teams = [team for group in wc_groups.values() for team in group]
field_elos = {team: elo_model.get_rating(team) for team in all_teams}

# ==========================================
# 4. INTERACTIVE DASHBOARD TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Full Tournament Simulation", "⚔️ Head-to-Head Matchup Arena", "📋 Sample Group Stage"])

# ----------------- TAB 1: MONTE CARLO TOURNAMENT -----------------
with tab1:
    st.write("Click below to run a Monte Carlo simulation (Group Stage + 32-Team Knockout).")
    if st.button("🚀 Run Cloud Tournament Simulation", type="primary"):
        with st.spinner(f"🎲 Simulating {iterations:,} tournaments... this takes about 5-10 seconds..."):
            hosts = ['United States', 'Mexico', 'Canada'] if host_advantage else []
            championship_counts = {team: 0 for team in all_teams}
            
            for _ in range(iterations):
                qualified_top2 = []
                third_place_pool = []
                for g_name, teams in wc_groups.items():
                    table = {t: [0, 0, 0, t] for t in teams} # points, gd, gf, name
                    for i in range(len(teams)):
                        for j in range(i + 1, len(teams)):
                            t1, t2 = teams[i], teams[j]
                            e1 = field_elos[t1] + (100 if t1 in hosts else 0)
                            e2 = field_elos[t2] + (100 if t2 in hosts else 0)
                            g1 = np.random.poisson(baseline_goals * np.exp((e1 - e2) / 400))
                            g2 = np.random.poisson(baseline_goals * np.exp((e2 - e1) / 400))
                            
                            table[t1][1] += (g1 - g2)
                            table[t1][2] += g1
                            table[t2][1] += (g2 - g1)
                            table[t2][2] += g2
                            
                            if g1 > g2: table[t1][0] += 3
                            elif g2 > g1: table[t2][0] += 3
                            else:
                                table[t1][0] += 1
                                table[t2][0] += 1
                    
                    sorted_group = sorted(table.values(), key=lambda x: (x[0], x[1], x[2]), reverse=True)
                    qualified_top2.extend([sorted_group[0][3], sorted_group[1][3]])
                    third_place_pool.append(sorted_group[2])
                    
                best_thirds = sorted(third_place_pool, key=lambda x: (x[0], x[1], x[2]), reverse=True)[:8]
                round_of_32 = qualified_top2 + [x[3] for x in best_thirds]
                
                current_round = round_of_32
                while len(current_round) > 1:
                    next_round = []
                    for i in range(0, len(current_round), 2):
                        t1, t2 = current_round[i], current_round[i+1]
                        e1 = field_elos[t1] + (100 if t1 in hosts else 0)
                        e2 = field_elos[t2] + (100 if t2 in hosts else 0)
                        g1 = np.random.poisson(baseline_goals * np.exp((e1 - e2) / 400))
                        g2 = np.random.poisson(baseline_goals * np.exp((e2 - e1) / 400))
                        winner = t1 if g1 > g2 else (t2 if g2 > g1 else (t1 if random.random() < (e1/(e1+e2)) else t2))
                        next_round.append(winner)
                    current_round = next_round
                    
                championship_counts[current_round[0]] += 1
                
            # Build Chart Data
            results_df = pd.DataFrame([
                {
                    "Raw Team": team,
                    "Team": get_flag(team), 
                    "Win Probability (%)": (wins / iterations) * 100, 
                    "Elo Rating": int(field_elos[team])
                }
                for team, wins in championship_counts.items() if wins > 0
            ]).sort_values(by="Win Probability (%)", ascending=False).reset_index(drop=True)
            
            # Map host colors for Plotly
            color_discrete_map = {get_flag(t): '#e74c3c' if t in hosts else '#3498db' for t in results_df['Raw Team']}
            
            fig = px.bar(results_df.head(15), x="Win Probability (%)", y="Team", orientation='h',
                         color="Team", color_discrete_map=color_discrete_map, text="Win Probability (%)",
                         title=f"Top 15 Title Favorites ({iterations:,} Iterations)")
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside', showlegend=False)
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(results_df[["Team", "Win Probability (%)", "Elo Rating"]], hide_index=True, use_container_width=True)

# ----------------- TAB 2: HEAD TO HEAD -----------------
with tab2:
    st.subheader("Simulate a Single Knockout Match")
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Select Team A (Home/Neutral)", sorted(all_teams), index=all_teams.index("United States"), format_func=get_flag)
    with col2:
        team_b = st.selectbox("Select Team B (Away)", sorted(all_teams), index=all_teams.index("Argentina"), format_func=get_flag)
        
    if team_a != team_b:
        elo_a = field_elos[team_a] + (100 if host_advantage and team_a in ['United States', 'Mexico', 'Canada'] else 0)
        elo_b = field_elos[team_b] + (100 if host_advantage and team_b in ['United States', 'Mexico', 'Canada'] else 0)
        
        lam_a = baseline_goals * np.exp((elo_a - elo_b) / 400)
        lam_b = baseline_goals * np.exp((elo_b - elo_a) / 400)
        
        st.write(f"**Expected Goals (xG):** {get_flag(team_a)} **{lam_a:.2f}** — **{lam_b:.2f}** {get_flag(team_b)}")
        
        sim_a = np.random.poisson(lam_a, size=10000)
        sim_b = np.random.poisson(lam_b, size=10000)
        
        wins_a = np.sum(sim_a > sim_b) / 100
        draws = np.sum(sim_a == sim_b) / 100
        wins_b = np.sum(sim_b > sim_a) / 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{get_flag(team_a)} Win", f"{wins_a:.1f}%")
        c2.metric("90-Min Draw", f"{draws:.1f}%")
        c3.metric(f"{get_flag(team_b)} Win", f"{wins_b:.1f}%")
        
        # Calculate Exact Scorelines
        st.divider()
        st.markdown("#### Most Likely Exact Scorelines (90 Mins)")
        scores_df = pd.DataFrame({'A': sim_a, 'B': sim_b})
        scores_df['Score'] = scores_df['A'].astype(str) + " - " + scores_df['B'].astype(str)
        top_scores = scores_df['Score'].value_counts(normalize=True).head(5) * 100
        
        sc_cols = st.columns(5)
        for idx, (score, prob) in enumerate(top_scores.items()):
            sc_cols[idx].metric(label=score, value=f"{prob:.1f}%")

# ----------------- TAB 3: GROUP STAGE -----------------
with tab3:
    st.write("Simulate exactly ONE instance of the group stage to see how points, goals, and tiebreakers play out!")
    if st.button("🎲 Simulate One Group Stage Draw", type="primary"):
        hosts = ['United States', 'Mexico', 'Canada'] if host_advantage else []
        grid_cols = st.columns(3)
        col_idx = 0
        
        for g_name, teams in wc_groups.items():
            table = {t: [0, 0, 0, 0, 0, t] for t in teams} # pts, gd, gf, ga, matches, name
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    t1, t2 = teams[i], teams[j]
                    e1 = field_elos[t1] + (100 if t1 in hosts else 0)
                    e2 = field_elos[t2] + (100 if t2 in hosts else 0)
                    g1 = np.random.poisson(baseline_goals * np.exp((e1 - e2) / 400))
                    g2 = np.random.poisson(baseline_goals * np.exp((e2 - e1) / 400))
                    
                    table[t1][4] += 1; table[t2][4] += 1
                    table[t1][2] += g1; table[t2][2] += g2
                    table[t1][3] += g2; table[t2][3] += g1
                    table[t1][1] += (g1 - g2); table[t2][1] += (g2 - g1)
                    
                    if g1 > g2: table[t1][0] += 3
                    elif g2 > g1: table[t2][0] += 3
                    else: table[t1][0] += 1; table[t2][0] += 1
            
            sorted_g = sorted(table.values(), key=lambda x: (x[0], x[1], x[2]), reverse=True)
            df_group = pd.DataFrame(sorted_g, columns=["Pts", "GD", "GF", "GA", "MP", "Team"])
            df_group['Team'] = df_group['Team'].apply(get_flag)
            df_group = df_group[["Team", "MP", "Pts", "GD", "GF", "GA"]]
            
            with grid_cols[col_idx % 3]:
                st.markdown(f"**{g_name}**")
                st.dataframe(df_group, hide_index=True)
                st.write("") # spacing
            col_idx += 1
