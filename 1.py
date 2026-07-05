import pandas as pd
import numpy as np
from scipy.stats import poisson
import random
import os

# ==========================================
# 1. THE ELO RATING SYSTEM
# ==========================================
class EloSystem:
    def __init__(self, default_elo=1500, k_factor=40):
        self.ratings = {}
        self.default_elo = default_elo
        self.k_factor = k_factor

    def get_rating(self, team):
        if team not in self.ratings:
            self.ratings[team] = self.default_elo
        return self.ratings[team]

    def compute_expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, team_a, team_b, goal_a, goal_b, is_neutral=False):
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        
        # Add home advantage adjustment if not a neutral venue match
        home_adv = 100 if not is_neutral else 0
        rating_a_adj = rating_a + home_adv
        
        # Expected outcomes (probabilities)
        ea = self.compute_expected_score(rating_a_adj, rating_b)
        eb = 1 - ea
        
        # Actual outcomes
        if goal_a > goal_b:
            wa, wb = 1, 0
        elif goal_a < goal_b:
            wa, wb = 0, 1
        else:
            wa, wb = 0.5, 0.5
            
        # Goal margin multiplier to reward decisive wins
        margin = abs(goal_a - goal_b)
        g_multiplier = 1 if margin <= 1 else (1.5 if margin == 2 else (11 + margin) / 8)
        
        # Calculate new ratings
        self.ratings[team_a] = rating_a + self.k_factor * g_multiplier * (wa - ea)
        self.ratings[team_b] = rating_b + self.k_factor * g_multiplier * (wb - eb)


# ==========================================
# 2. THE POISSON GOALS PREDICTOR
# ==========================================
class PoissonPredictor:
    def __init__(self, baseline_goals=1.35):
        self.baseline_goals = baseline_goals

    def calculate_lambdas(self, elo_a, elo_b, is_neutral=True):
        # Home advantage baseline for non-neutral games
        elo_a_adj = elo_a + (50 if not is_neutral else 0)
        
        # Scale Elo differences to expected goals (lambda)
        lambda_a = self.baseline_goals * np.exp((elo_a_adj - elo_b) / 400)
        lambda_b = self.baseline_goals * np.exp((elo_b - elo_a_adj) / 400)
        
        return lambda_a, lambda_b


# ==========================================
# 3. MONTE CARLO TOURNAMENT SIMULATOR
# ==========================================
class TournamentSimulator:
    def __init__(self, predictor):
        self.predictor = predictor

    def simulate_match_knockout(self, team_a, elo_a, team_b, elo_b):
        # Compute individual expected goal rates (lambda)
        lam_a, lam_b = self.predictor.calculate_lambdas(elo_a, elo_b, is_neutral=True)
        
        # Sample actual goals using a Poisson distribution
        goals_a = np.random.poisson(lam_a)
        goals_b = np.random.poisson(lam_b)
        
        if goals_a > goals_b:
            return team_a
        elif goals_b > goals_a:
            return team_b
        else:
            # Tiebreaker (Extra Time / Penalties) modeled via original Elo win probability
            elo_total = elo_a + elo_b
            return team_a if random.random() < (elo_a / elo_total) else team_b

    def simulate_bracket(self, round_16_teams, elos):
        """Simulates a standard knockout structure from the Round of 16 down to the winner."""
        current_round = list(round_16_teams)
        
        while len(current_round) > 1:
            next_round = []
            for i in range(0, len(current_round), 2):
                t1, t2 = current_round[i], current_round[i+1]
                winner = self.simulate_match_knockout(t1, elos[t1], t2, elos[t2])
                next_round.append(winner)
            current_round = next_round
            
        return current_round[0]


# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    csv_filename = "results.csv"
    
    # Check if data file exists
    if not os.path.exists(csv_filename):
        print(f"Error: Could not find '{csv_filename}' in your current directory.")
        print("Please download it from the GitHub repository and save it in this folder.")
        exit()

    print("Step 1: Loading historical international match results...")
    df = pd.read_csv(csv_filename)
    
    # Basic data cleaning
    df = df.dropna(subset=['home_score', 'away_score', 'home_team', 'away_team'])
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    
    print(f"Step 2: Training Elo ratings on {len(df):,} historical matches...")
    elo = EloSystem()
    for _, row in df.iterrows():
        elo.update_ratings(
            row['home_team'], row['away_team'], 
            row['home_score'], row['away_score'], 
            is_neutral=row['neutral']
        )
        
    # Define a realistic knockout stage layout (Round of 16)
    wc_field = [
        'Argentina', 'France',    # Matchup 1
        'Brazil', 'England',       # Matchup 2
        'Spain', 'Germany',        # Matchup 3
        'Netherlands', 'Portugal', # Matchup 4
        'Morocco', 'Japan',        # Matchup 5
        'Croatia', 'Italy',        # Matchup 6
        'USA', 'Mexico',           # Matchup 7
        'Senegal', 'Colombia'      # Matchup 8
    ]
    
    # Retrieve final trained Elos for our teams
    field_elos = {team: elo.get_rating(team) for team in wc_field}
    
    print("\nTrained Team Elo Ratings:")
    for team, rating in sorted(field_elos.items(), key=lambda x: x[1], reverse=True):
        print(f" - {team}: {int(rating)}")

    # Initialize simulation models
    predictor = PoissonPredictor()
    simulator = TournamentSimulator(predictor)
    
    # Run Monte Carlo loop
    iterations = 10000
    championship_counts = {team: 0 for team in wc_field}
    
    print(f"\nStep 3: Running {iterations:,} Monte Carlo simulations of the knockout stage...")
    for _ in range(iterations):
        winner = simulator.simulate_bracket(wc_field, field_elos)
        championship_counts[winner] += 1
        
    print("\n=========================================")
    print("      WORLD CUP WIN PROBABILITIES        ")
    print("=========================================")
    for team, wins in sorted(championship_counts.items(), key=lambda item: item[1], reverse=True):
        probability = (wins / iterations) * 100
        print(f"{team:<15} | Probability: {probability:>6.2f}% | Elo: {int(field_elos[team])}")
    print("=========================================")