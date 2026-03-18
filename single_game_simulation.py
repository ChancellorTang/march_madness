import pandas as pd
import argparse
import sys  
from datetime import datetime
import joblib

current_year = datetime.now().year

parser = argparse.ArgumentParser(description='Simulate a single NCAA tournament game between two teams.')
parser.add_argument('team1', type=str, help='First team name')
parser.add_argument('team2', type=str, help='Second team name')
parser.add_argument('--year', type=int, help='Year to process', default=current_year)
parser.add_argument('--model', type=str, help='Model for master sim_type', default='knn')
parser.add_argument('--w1_model', type=str, help='W1 model for weeks sim_type', default='knn')
parser.add_argument('--w2_model', type=str, help='W2 model for weeks sim_type', default='knn')
parser.add_argument('--ff_model', type=str, help='FF model for weeks sim_type', default='knn')
parser.add_argument('--big_model', type=str, help='Big model for seed_diff sim_type', default='knn')
parser.add_argument('--little_model', type=str, help='Little model for seed_diff sim_type', default='knn')
parser.add_argument('--comp_model', type=str, help='Comp model for seed_diff sim_type', default='knn')
parser.add_argument('--sim_type', type=str, help='Type of simulation to run', default='master', choices=['master', 'weeks', 'seed_diff'])
args = parser.parse_args()

model_types = ['knn', 'DT', 'forest', 'mlp', 'clf', 'gnb', 'svc']

scaler = joblib.load('models/my_scaler.pkl')

sim_type = args.sim_type

# Validate models based on sim_type
def validate_models(sim_type, args):
    """Validate that specified models exist for the given sim_type."""
    if sim_type == "master":
        if args.model not in model_types:
            sys.exit(f"Model '{args.model}' not recognized. Please choose from: {', '.join(model_types)}")
    elif sim_type == "weeks":
        for model_name in [args.w1_model, args.w2_model, args.ff_model]:
            if model_name not in model_types:
                sys.exit(f"Model '{model_name}' not recognized. Please choose from: {', '.join(model_types)}")
    elif sim_type == "seed_diff":
        for model_name in [args.big_model, args.little_model, args.comp_model]:
            if model_name not in model_types:
                sys.exit(f"Model '{model_name}' not recognized. Please choose from: {', '.join(model_types)}")

validate_models(sim_type, args)

match sim_type:
    case "master":
        master = joblib.load('models/{}/master.pkl'.format(args.model))
    case "weeks":
        w1 = joblib.load('models/{}/w1.pkl'.format(args.w1_model))
        w2 = joblib.load('models/{}/w2.pkl'.format(args.w2_model))
        ff = joblib.load('models/{}/ff.pkl'.format(args.ff_model))
    case "seed_diff":
        big = joblib.load('models/{}/big.pkl'.format(args.big_model))
        little = joblib.load('models/{}/little.pkl'.format(args.little_model))
        comp = joblib.load('models/{}/comp.pkl'.format(args.comp_model))
        seed_cutoff_high = -4
        seed_cutoff_low = -7

def scale(df):
    m = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return m

def get_upset_differences(a, b):
    """
    Takes two rows from one of the region databases and finds the difference between the two columns 
    a: higher seed
    b: lower seed
    output: list of db difference rows
    """
    listA = []
    for x in range(3, 25):
        diff = a.iloc[x] - b.iloc[x]
        listA.append(diff)
    return listA

master_df = pd.read_csv("data/first_four/first_four26.csv")

df_headers = ['TEAM',
 'CONF',
 'POSTSEASON',
 'G',
 'W',
 'WIN_PER',
 'ADJOE',
 'ADJDE',
 'BARTHAG',
 'EFG_O',
 'EFG_D',
 'TOR',
 'TORD',
 'ORB',
 'DRB',
 'FTR',
 'FTRD',
 '2P_O',
 '2P_D',
 '3P_O',
 '3P_D',
 'ADJ_T',
 'WAB',
 'SEED',
 'POWER',
 'YEAR']

train_columns = df_headers[3:25]

def seed_diff_sim(holder, scaled, round):
    if holder.iloc[0]["SEED"] < seed_cutoff_high:
        ups = big.predict(scaled)
    elif holder.iloc[0]["SEED"] > seed_cutoff_low:
        ups = little.predict(scaled)
    else:
        ups = comp.predict(scaled)
    return int(ups[0])

def master_sim(holder, scaled, round):
    return int(master.predict(scaled)[0])

def weeks_sim(holder, scaled, round):
    if round in [0, 1]:
        return int(w1.predict(scaled)[0])
    elif round in [2, 3]:
        return int(w2.predict(scaled)[0])
    else:
        return int(ff.predict(scaled)[0])

def play_one_match(h, l, round, sim=sim_type):
    """Return (winner_row, holder_df, matchup_str, prediction_int)."""
    holder = pd.DataFrame([get_upset_differences(h, l)], columns=train_columns)
    # ensure holder is oriented so that lower SEED corresponds to label 1 as before
    if holder.iloc[0]["SEED"] > 0:
        holder = -holder
        h, l = l, h
    scaled = scale(holder)
    pred = globals()[f"{sim}_sim"](holder, scaled, round)
    winner = h if pred == 0 else l
    matchup = f"{h['TEAM']} vs. {l['TEAM']}"
    return winner, holder, matchup, pred

# Look up teams
team1_name = args.team1
team2_name = args.team2

team1_data = master_df[(master_df.TEAM.str.lower() == team1_name.lower()) & (master_df.YEAR == args.year)]
team2_data = master_df[(master_df.TEAM.str.lower() == team2_name.lower()) & (master_df.YEAR == args.year)]

if team1_data.empty:
    sys.exit(f"Team '{team1_name}' not found for year {args.year}")
if team2_data.empty:
    sys.exit(f"Team '{team2_name}' not found for year {args.year}")

team1_row = team1_data.iloc[0]
team2_row = team2_data.iloc[0]

# Determine higher vs lower seed
if team1_row['SEED'] <= team2_row['SEED']:
    higher_seed = team1_row
    lower_seed = team2_row
else:
    higher_seed = team2_row
    lower_seed = team1_row

# Run simulation
winner, holder, matchup, pred = play_one_match(higher_seed, lower_seed, round=0, sim=sim_type)

# Display results
print("\n" + "="*60)
print("SINGLE GAME SIMULATION")
print("="*60)
print(f"Year: {args.year}")
print(f"Sim Type: {sim_type}")
print(f"Model: {args.model if sim_type == 'master' else 'Multiple'}")
print("="*60)
print(f"\n{higher_seed['SEED']} (Higher Seed) {higher_seed['TEAM']}")
print(f"vs.")
print(f"{lower_seed['SEED']} (Lower Seed) {lower_seed['TEAM']}\n")
print(f"WINNER: {winner['SEED']} {winner['TEAM']}")
print("="*60 + "\n")

if winner['SEED'] < higher_seed['SEED']:
    print(f"✓ (Expected) Higher seed {higher_seed['SEED']} {higher_seed['TEAM']} wins")
elif winner['SEED'] > higher_seed['SEED']:
    print(f"⚠ (Upset!) Lower seed {winner['SEED']} {winner['TEAM']} defeats {higher_seed['SEED']} {higher_seed['TEAM']}")
else:    
    print(f"Winner: {winner['TEAM']}")

print()
