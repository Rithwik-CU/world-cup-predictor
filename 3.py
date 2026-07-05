import pandas as pd
import numpy as np
from scipy.stats import poisson
import random
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. ADVANCED ELO WITH DYNAMIC K-FACTORS
# ==========================================
class EloSystem:
    def __init__(self, default_elo=1500):
        self.ratings = {}
        self.default_elo = default_elo

    def get_rating(self, team):
        if team == 'USA': team = 'United States'
        if team not in self.ratings:
            self.ratings[team] = self.default_elo
        return self.ratings[team]

    def get_k_factor(self, tournament_name):
        t_str = str(tournament_name).lower()
        if 'world cup' in t_str and 'qualification' not in t_str:
            return 60
        elif 'qualification' in t_str or 'copa america' in t_str or 'euro' in t_str:
            return 40
        else:
            return 20

    def update_ratings(self, team_a, team_b, goal_a, goal_b, tournament, is_neutral=False):
        if team_a == 'USA': team_a = 'United States'
        if team_b == 'USA': team_b = 'United States'
        
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        k_factor = self.get_k_factor(tournament)
        
        home_adv = 100 if not is_neutral else 0
        ea = 1 / (1 + 10 ** (((rating_b - (rating_a + home_adv))) / 400))
        eb = 1 - ea
        
        if goal_a > goal_b: wa, wb = 1, 0
        elif goal_a < goal_b: wa, wb = 0, 1
        else: wa, wb = 0.5, 0.5
            
        margin = abs(goal_a - goal_b)
        g_mult = 1 if margin <= 1 else (1.5 if margin == 2 else (11 + margin) / 8)
        
        self.ratings[team_a] = rating_a + k_factor * g_mult * (wa - ea)
        self.ratings[team_b] = rating_b + k_factor * g_mult * (wb - eb)


# ==========================================
# 2. POISSON PREDICTOR WITH HOST ADVANTAGE
# ==========================================
class PoissonPredictor:
    def __init__(self, baseline_goals=1.35):
        self.baseline_goals = baseline_goals
        # Define the three official co-hosts of the 2026 World Cup
        self.hosts = ['United States', 'Mexico', 'Canada']

    def calculate_lambdas(self, elo_a, team_a, elo_b, team_b):
        """Calculates expected goals, adding an Elo boost if a team is a tournament host."""
        elo_a_adj = elo_a
        elo_b_adj = elo_b

        # Apply host nation crowd/familiarity advantage (+100 Elo strength boost)
        if team_a in self.hosts:
            elo_a_adj += 100
        if team_b in self.hosts:
            elo_b_adj += 100

        lambda_a = self.baseline_goals * np.exp((elo_a_adj - elo_b_adj) / 400)
        lambda_b = self.baseline_goals * np.exp((elo_b_adj - elo_a_adj) / 400)
        return lambda_a, lambda_b


# ==========================================
# 3. 48-TEAM TOURNAMENT SIMULATOR
# ==========================================
class TournamentSimulator:
    def __init__(self, predictor):
        self.predictor = predictor

    def simulate_match_goals(self, team_a, elo_a, team_b, elo_b):
        lam_a, lam_b = self.predictor.calculate_lambdas(elo_a, team_a, elo_b, team_b)
        return np.random.poisson(lam_a), np.random.poisson(lam_b)

    def simulate_group_stage(self, groups, elos):
        qualified_top2 = []
        third_place_pool = []

        for group_name, teams in groups.items():
            table = {t: [0, 0, 0, t] for t in teams}
            
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    t1, t2 = teams[i], teams[j]
                    g1, g2 = self.simulate_match_goals(t1, elos[t1], t2, elos[t2])
                    
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
            qualified_top2.append(sorted_group[0][3])
            qualified_top2.append(sorted_group[1][3])
            third_place_pool.append(sorted_group[2])

        best_thirds = sorted(third_place_pool, key=lambda x: (x[0], x[1], x[2]), reverse=True)[:8]
        return qualified_top2 + [item[3] for item in best_thirds]

    def simulate_knockout_bracket(self, teams_32, elos):
        current_round = list(teams_32)
        while len(current_round) > 1:
            next_round = []
            for i in range(0, len(current_round), 2):
                t1, t2 = current_round[i], current_round[i+1]
                g1, g2 = self.simulate_match_goals(t1, elos[t1], t2, elos[t2])
                
                if g1 > g2: winner = t1
                elif g2 > g1: winner = t2
                else:
                    winner = t1 if random.random() < (elos[t1] / (elos[t1] + elos[t2])) else t2
                next_round.append(winner)
            current_round = next_round
        return current_round[0]


# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    csv_filename = "results.csv"
    if not os.path.exists(csv_filename):
        print("Error: Could not find 'results.csv'.")
        exit()

    print("Step 1: Training Advanced Elo on historical data...")
    df = pd.read_csv(csv_filename).dropna(subset=['home_score', 'away_score', 'home_team', 'away_team'])
    
    elo = EloSystem()
    for _, row in df.iterrows():
        elo.update_ratings(row['home_team'], row['away_team'], int(row['home_score']), int(row['away_score']), row.get('tournament', 'Friendly'), is_neutral=row['neutral'])
        
    wc_groups = {
        'Group A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
        'Group B': ['Canada', 'Switzerland', 'Qatar', 'Bosnia and Herzegovina'],
        'Group C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
        'Group D': ['USA', 'Paraguay', 'Australia', 'Turkey'],
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
    field_elos = {team: elo.get_rating(team) for team in all_teams}

    predictor = PoissonPredictor()
    simulator = TournamentSimulator(predictor)
    
    iterations = 5000
    championship_counts = {team: 0 for team in all_teams}
    
    print(f"\nStep 2: Running {iterations:,} simulations with Host Advantage enabled...")
    for _ in range(iterations):
        advancing_32 = simulator.simulate_group_stage(wc_groups, field_elos)
        winner = simulator.simulate_knockout_bracket(advancing_32, field_elos)
        championship_counts[winner] += 1
        
    print("\n=========================================")
    print(" 2026 WORLD CUP WIN PROBABILITIES (TOP 15)")
    print("=========================================")
    top_15 = sorted(championship_counts.items(), key=lambda item: item[1], reverse=True)[:15]
    for team, wins in top_15:
        probability = (wins / iterations) * 100
        print(f"{team:<15} | Probability: {probability:>6.2f}% | Elo: {int(field_elos[team])}")
    print("=========================================")

    # ==========================================
    # 5. VISUAL CHART GENERATION
    # ==========================================
    print("\nStep 3: Generating and exporting high-resolution data visualization...")
    
    # Structure data for plotting (reverse lists so highest value appears at the top of the horizontal bar chart)
    teams_plot = [x[0] for x in top_15][::-1]
    probs_plot = [(x[1] / iterations) * 100 for x in top_15][::-1]
    
    plt.figure(figsize=(12, 7))
    
    # Custom conditional coloring: highlight host nations if they make the top 15
    bar_colors = []
    for team in teams_plot:
        if team in ['United States', 'Mexico', 'Canada']:
            bar_colors.append('#e74c3c') # Soft red for hosts
        else:
            bar_colors.append('#34495e') # Slate grey for regular teams
            
    bars = plt.barh(teams_plot, probs_plot, color=bar_colors, edgecolor='none', height=0.6)
    
    # Formatting chart presentation
    plt.xlabel('Championship Probability (%)', fontsize=11, fontweight='bold', labelpad=10)
    plt.title('2026 World Cup Win Probabilities\n(Derived from 5,000 Monte Carlo Simulations)', fontsize=14, fontweight='bold', pad=15)
    plt.xlim(0, max(probs_plot) + 4) # Add buffer space to prevent right-edge clipping
    
    # Clean grid lines
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.gca().set_axisbelow(True) 
    
    # Remove outer box borders for a modern design look
    for spine in ['top', 'right', 'bottom', 'left']:
        plt.gca().spines[spine].set_visible(False)

    # Attach precise numeric labels to the end of each bar
    for bar in bars:
        width = bar.get_width()
        if width > 0:
            plt.text(width + 0.3, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', 
                     va='center', ha='left', fontsize=10, fontweight='bold')
                     
    plt.tight_layout()
    
    # Save chart to local project folder
    output_img = 'world_cup_odds.png'
    plt.savefig(output_img, dpi=300)
    plt.close()
    
    print(f"Success! Visual chart saved to your workspace as '{output_img}'")