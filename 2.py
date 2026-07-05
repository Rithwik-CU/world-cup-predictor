import pandas as pd
import numpy as np
from scipy.stats import poisson
import random
import os

# ==========================================
# 1. ADVANCED ELO WITH DYNAMIC K-FACTORS
# ==========================================
class EloSystem:
    def __init__(self, default_elo=1500):
        self.ratings = {}
        self.default_elo = default_elo

    def get_rating(self, team):
        # Automatically map USA to official dataset name
        if team == 'USA': team = 'United States'
        if team not in self.ratings:
            self.ratings[team] = self.default_elo
        return self.ratings[team]

    def get_k_factor(self, tournament_name):
        t_str = str(tournament_name).lower()
        if 'world cup' in t_str and 'qualification' not in t_str:
            return 60  # Major Tournament Finals
        elif 'qualification' in t_str or 'copa america' in t_str or 'euro' in t_str:
            return 40  # Competitive Qualifiers & Continental Cups
        else:
            return 20  # Friendlies & Minor Tournaments

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

    def compute_expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


# ==========================================
# 2. POISSON GOALS PREDICTOR
# ==========================================
class PoissonPredictor:
    def __init__(self, baseline_goals=1.35):
        self.baseline_goals = baseline_goals

    def calculate_lambdas(self, elo_a, elo_b, is_neutral=True):
        elo_a_adj = elo_a + (50 if not is_neutral else 0)
        lambda_a = self.baseline_goals * np.exp((elo_a_adj - elo_b) / 400)
        lambda_b = self.baseline_goals * np.exp((elo_b - elo_a_adj) / 400)
        return lambda_a, lambda_b


# ==========================================
# 3. 48-TEAM TOURNAMENT SIMULATOR
# ==========================================
class TournamentSimulator:
    def __init__(self, predictor):
        self.predictor = predictor

    def simulate_match_goals(self, team_a, elo_a, team_b, elo_b):
        lam_a, lam_b = self.predictor.calculate_lambdas(elo_a, elo_b, is_neutral=True)
        return np.random.poisson(lam_a), np.random.poisson(lam_b)

    def simulate_group_stage(self, groups, elos):
        """Simulates 12 groups of 4 teams. Returns the 32 teams advancing to knockouts."""
        qualified_top2 = []
        third_place_pool = []

        for group_name, teams in groups.items():
            # Track: [Points, Goal Difference, Goals For, Team Name]
            table = {t: [0, 0, 0, t] for t in teams}
            
            # Round-robin: every team plays each other once
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    t1, t2 = teams[i], teams[j]
                    g1, g2 = self.simulate_match_goals(t1, elos[t1], t2, elos[t2])
                    
                    # Update Goals
                    table[t1][1] += (g1 - g2)
                    table[t1][2] += g1
                    table[t2][1] += (g2 - g1)
                    table[t2][2] += g2
                    
                    # Update Points
                    if g1 > g2: table[t1][0] += 3
                    elif g2 > g1: table[t2][0] += 3
                    else:
                        table[t1][0] += 1
                        table[t2][0] += 1
                        
            # Sort group by Points, then Goal Difference, then Goals For
            sorted_group = sorted(table.values(), key=lambda x: (x[0], x[1], x[2]), reverse=True)
            
            # Top 2 advance automatically
            qualified_top2.append(sorted_group[0][3])
            qualified_top2.append(sorted_group[1][3])
            # Save 3rd place for wild-card ranking
            third_place_pool.append(sorted_group[2])

        # Rank all 3rd place teams across groups and take the best 8
        best_thirds = sorted(third_place_pool, key=lambda x: (x[0], x[1], x[2]), reverse=True)[:8]
        qualified_thirds = [item[3] for item in best_thirds]
        
        return qualified_top2 + qualified_thirds

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
                    # Penalties tiebreaker based on original Elo win probability
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
        print("Error: Could not find 'results.csv'. Please place it in this folder.")
        exit()

    print("Step 1: Training Advanced Elo (with Dynamic K-Factors) on historical data...")
    df = pd.read_csv(csv_filename).dropna(subset=['home_score', 'away_score', 'home_team', 'away_team'])
    df['home_score'], df['away_score'] = df['home_score'].astype(int), df['away_score'].astype(int)
    
    elo = EloSystem()
    for _, row in df.iterrows():
        elo.update_ratings(
            row['home_team'], row['away_team'], 
            row['home_score'], row['away_score'], 
            row.get('tournament', 'Friendly'), 
            is_neutral=row['neutral']
        )
        
    # 2026 World Cup Sample Field (12 Groups of 4 Teams = 48 Teams)
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
    
    # Retrieve final trained Elos (mapped automatically for USA -> United States)
    all_teams = [team for group in wc_groups.values() for team in group]
    field_elos = {team: elo.get_rating(team) for team in all_teams}
    
    print("\nTop 10 Trained Team Elo Ratings in the Tournament:")
    for team, rating in sorted(field_elos.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f" - {team}: {int(rating)}")

    predictor = PoissonPredictor()
    simulator = TournamentSimulator(predictor)
    
    iterations = 5000  # 5,000 runs of a 48-team tournament
    championship_counts = {team: 0 for team in all_teams}
    
    print(f"\nStep 2: Running {iterations:,} simulations of the 48-team Group Stage & Knockouts...")
    for _ in range(iterations):
        # 1. Run round-robin group stage to get top 32 teams
        advancing_32 = simulator.simulate_group_stage(wc_groups, field_elos)
        # 2. Run knockout bracket to get champion
        winner = simulator.simulate_knockout_bracket(advancing_32, field_elos)
        championship_counts[winner] += 1
        
    print("\n=========================================")
    print(" 2026 WORLD CUP WIN PROBABILITIES (TOP 15)")
    print("=========================================")
    for team, wins in sorted(championship_counts.items(), key=lambda item: item[1], reverse=True)[:15]:
        probability = (wins / iterations) * 100
        print(f"{team:<15} | Probability: {probability:>6.2f}% | Elo: {int(field_elos[team])}")
    print("=========================================")