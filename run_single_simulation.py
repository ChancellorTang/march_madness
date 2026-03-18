import pandas as pd
import argparse
import sys  
from datetime import datetime
import numpy as np
import joblib
import json

current_year = datetime.now().year

parser = argparse.ArgumentParser(description='Run NCAA tournament simulation with model selection.')
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

scaler = joblib.load('models/my_scaler.pkl')  # load from disk

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
        master = joblib.load('models/{}/master.pkl'.format(args.model))  # load from disk
    case "weeks":
        w1 = joblib.load('models/{}/w1.pkl'.format(args.w1_model))  # load from disk
        w2 = joblib.load('models/{}/w2.pkl'.format(args.w2_model))  # load from disk
        ff = joblib.load('models/{}/ff.pkl'.format(args.ff_model))  # load from disk

    case "seed_diff":
        big = joblib.load('models/{}/big.pkl'.format(args.big_model))  # load from disk
        little = joblib.load('models/{}/little.pkl'.format(args.little_model))  # load from disk
        comp = joblib.load('models/{}/comp.pkl'.format(args.comp_model))  # load from disk
        seed_cutoff_high = -4
        seed_cutoff_low = -7


def scale(df):
    m = pd.DataFrame(scaler.transform(df), columns = df.columns)
    return m

def get_upset_differences(a,b):
    """
    Takes two rows from one of the region databases and finds the difference between the two columns 

    a: higher seed
    b: lower seed'

    output: list of db difference rows
    """
    
    listA = []
    for x in range(3,25):
        diff = a.iloc[x] - b.iloc[x]
        listA.append(diff)
    return listA

if args.model in model_types:
    model = args.model
else:
    sys.exit("Model not recognized. Please choose from: " + ", ".join(model_types))

args = parser.parse_args()

master_df = pd.read_csv("master_ncaa.csv")

args = parser.parse_args()

regions = ['east','south','west','midwest']

test_regions = []
for region in regions:
    test_regions.append(master_df[(master_df.REGION == region) & (master_df.YEAR == args.year)].sort_values('SEED'))

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

# Remaining Teams
r64_r = pd.DataFrame(columns = df_headers)
r32_r = pd.DataFrame(columns = df_headers)
s16_r = pd.DataFrame(columns = df_headers)
e8_r = pd.DataFrame(columns = df_headers)
f4_r = pd.DataFrame(columns = df_headers)
c2_r= pd.DataFrame(columns = df_headers)
winner_r= pd.DataFrame(columns = df_headers)

# test df
test_df = pd.DataFrame(columns = train_columns)
y_pred = []
matchup_list = []

def seed_diff_sim(holder, scaled, round):
    if holder.iloc[0]["SEED"] < seed_cutoff_high:
        ups = big.predict(scaled)
    elif  holder.iloc[0]["SEED"] > seed_cutoff_low:
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

# compacted implementation for rounds and finals
def play_one_match(h, l, round, sim = sim_type):
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


def get_filename_for_models(sim_type, args):
    """Generate filename based on sim_type and models used."""
    if sim_type == "master":
        return f"{args.model}_{sim_type}"
    elif sim_type == "weeks":
        return f"w1_{args.w1_model}_w2_{args.w2_model}_ff_{args.ff_model}_{sim_type}"
    elif sim_type == "seed_diff":
        return f"big_{args.big_model}_little_{args.little_model}_comp_{args.comp_model}_{sim_type}"


def run_rounds(start_df, n_rounds):
    """
    Run n_rounds knockout rounds starting from start_df.
    Returns list_of_rounds where list_of_rounds[0] == start_df,
    and subsequent entries are the winners of each round.
    Side-effects: appends to test_df, y_pred, matchup_list and prints results.
    """
    rounds = [start_df.reset_index(drop=True)]
    for _ in range(n_rounds):
        cur = rounds[-1]
        winners = []
        for i in range(len(cur) // 2):
            h = cur.iloc[i]
            l = cur.iloc[-i-1]
            winner, holder, matchup, pred = play_one_match(h, l, _, sim_type)
            winners.append(winner)
            # update global tracking structures
            globals()['test_df'] = pd.concat([globals()['test_df'], holder], ignore_index=True)
            globals()['y_pred'].append(pred)
            globals()['matchup_list'].append(matchup)
            print(f"{h['SEED']} {h['TEAM']}  vs.  {l['SEED']} {l['TEAM']}")
            print("Winner:", winner['SEED'], winner['TEAM'])
        rounds.append(pd.DataFrame(winners, columns=df_headers))
        print("\n")
    return rounds

# Remaining Teams (kept as empty frames from earlier)
# r64_r, r32_r, s16_r, e8_r, f4_r, c2_r, winner_r already defined above

# process each region (compact)
for region_df in test_regions:
    r64_r = pd.concat([r64_r, region_df], ignore_index=True)
    rounds = run_rounds(region_df, n_rounds=4)  # r64 -> r32 -> s16 -> e8 -> f4
    # rounds is a list: [r64, r32, s16, e8, f4]
    r32_r = pd.concat([r32_r, rounds[1]], ignore_index=True)
    s16_r = pd.concat([s16_r, rounds[2]], ignore_index=True)
    e8_r  = pd.concat([e8_r, rounds[3]], ignore_index=True)
    f4_r  = pd.concat([f4_r, rounds[4]], ignore_index=True)
    print("_" * 40)

now = int(datetime.now().timestamp())



final_rounds = run_rounds(f4_r, n_rounds=2)  # f4 -> c2 -> winner
c2_r = pd.concat([c2_r, final_rounds[1]], ignore_index=True)
winner_r = pd.concat([winner_r, final_rounds[2]], ignore_index=True)

sim_json = {}
for df in [r64_r, r32_r, s16_r, e8_r, f4_r, c2_r, winner_r]:
    if len(df) > 2:
        # Properly distribute teams across 4 regions
        teams_per_region = len(df) // 4
        result = []
        for i in range(len(df)):
            result.append(regions[i % 4])
        df['region'] = result
    else:
        df['region'] = None
    sim_json.update({str(len(df)): df[['SEED', "TEAM", "region"]].to_dict(orient="records")})



with open(f"./Sims/{current_year}/{get_filename_for_models(sim_type, args)}_{str(now)}.json", 'w') as f:
    json.dump(sim_json, f, indent=4)