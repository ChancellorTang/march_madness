import pandas as pd
import argparse
import sys
from datetime import datetime
import numpy as np
import joblib
import json
from itertools import product
from pathlib import Path

current_year = datetime.now().year

# ==================== Setup ====================

scaler = joblib.load('models/my_scaler.pkl')

# Available models and sim types
MODEL_TYPES = ['knn', 'DT', 'forest', 'mlp', 'clf', 'gnb', 'svc']
REGIONS = ['east', 'south', 'west', 'midwest']

DF_HEADERS = ['TEAM', 'CONF', 'POSTSEASON', 'G', 'W', 'WIN_PER', 'ADJOE', 'ADJDE', 
              'BARTHAG', 'EFG_O', 'EFG_D', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', 
              '2P_O', '2P_D', '3P_O', '3P_D', 'ADJ_T', 'WAB', 'SEED', 'POWER', 'YEAR']

TRAIN_COLUMNS = DF_HEADERS[3:25]


# ==================== Permutation Generators ====================

def generate_master_permutations():
    """Yields: (model,) for master sim_type."""
    for model in MODEL_TYPES:
        yield (model,)


def generate_weeks_permutations():
    """Yields: (w1_model, w2_model, ff_model) for weeks sim_type."""
    for combo in product(MODEL_TYPES, repeat=3):
        yield combo


def generate_seed_diff_permutations():
    """Yields: (big_model, little_model, comp_model) for seed_diff sim_type."""
    for combo in product(MODEL_TYPES, repeat=3):
        yield combo


def get_permutations(sim_type):
    """Return the appropriate permutation generator for sim_type."""
    if sim_type == "master":
        return generate_master_permutations()
    elif sim_type == "weeks":
        return generate_weeks_permutations()
    elif sim_type == "seed_diff":
        return generate_seed_diff_permutations()
    else:
        raise ValueError(f"Unknown sim_type: {sim_type}")


# ==================== Model Loading ====================

def load_models_for_combo(sim_type, model_combo):
    """
    Load model pkl files based on sim_type and model_combo.
    
    Args:
        sim_type: 'master', 'weeks', or 'seed_diff'
        model_combo: tuple of model names
    
    Returns:
        dict: loaded model objects (keys depend on sim_type)
    """
    models = {}
    
    if sim_type == "master":
        model = model_combo[0]
        models['master'] = joblib.load(f'models/{model}/master.pkl')
        
    elif sim_type == "weeks":
        w1_model, w2_model, ff_model = model_combo
        models['w1'] = joblib.load(f'models/{w1_model}/w1.pkl')
        models['w2'] = joblib.load(f'models/{w2_model}/w2.pkl')
        models['ff'] = joblib.load(f'models/{ff_model}/ff.pkl')
        
    elif sim_type == "seed_diff":
        big_model, little_model, comp_model = model_combo
        models['big'] = joblib.load(f'models/{big_model}/big.pkl')
        models['little'] = joblib.load(f'models/{little_model}/little.pkl')
        models['comp'] = joblib.load(f'models/{comp_model}/comp.pkl')
    
    return models


# ==================== Helper Functions ====================

def scale(df):
    """Scale dataframe using the pre-loaded scaler."""
    m = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return m


def get_upset_differences(a, b):
    """
    Takes two rows and finds the difference between columns 3-25.
    
    Args:
        a: higher seed row
        b: lower seed row
    
    Returns:
        list of differences
    """
    listA = []
    for x in range(3, 25):
        diff = a.iloc[x] - b.iloc[x]
        listA.append(diff)
    return listA


def play_one_match(h, l, round_num, models, sim_type, seed_cutoff_high=-4, seed_cutoff_low=-7):
    """
    Play a single match and return winner.
    
    Args:
        h: higher seed row
        l: lower seed row
        round_num: round number (0-indexed)
        models: dict of loaded models
        sim_type: 'master', 'weeks', or 'seed_diff'
        seed_cutoff_high: cutoff for seed_diff (high seeds)
        seed_cutoff_low: cutoff for seed_diff (low seeds)
    
    Returns:
        winner, holder_df, matchup_str, prediction_int
    """
    holder = pd.DataFrame([get_upset_differences(h, l)], columns=TRAIN_COLUMNS)
    
    # Orient so lower SEED corresponds to label 1
    if holder.iloc[0]["SEED"] > 0:
        holder = -holder
        h, l = l, h
    
    scaled = scale(holder)
    
    # Get prediction based on sim_type
    if sim_type == "master":
        pred = int(models['master'].predict(scaled)[0])
    elif sim_type == "weeks":
        if round_num in [0, 1]:
            pred = int(models['w1'].predict(scaled)[0])
        elif round_num in [2, 3]:
            pred = int(models['w2'].predict(scaled)[0])
        else:
            pred = int(models['ff'].predict(scaled)[0])
    elif sim_type == "seed_diff":
        if holder.iloc[0]["SEED"] < seed_cutoff_high:
            pred = int(models['big'].predict(scaled)[0])
        elif holder.iloc[0]["SEED"] > seed_cutoff_low:
            pred = int(models['little'].predict(scaled)[0])
        else:
            pred = int(models['comp'].predict(scaled)[0])
    
    winner = h if pred == 0 else l
    matchup = f"{h['TEAM']} vs. {l['TEAM']}"
    
    return winner, holder, matchup, pred


def run_rounds(start_df, n_rounds, models, sim_type):
    """
    Run n_rounds knockout rounds starting from start_df.
    
    Args:
        start_df: starting dataframe of teams
        n_rounds: number of rounds to play
        models: dict of loaded models
        sim_type: 'master', 'weeks', or 'seed_diff'
    
    Returns:
        list of rounds [r64, r32, s16, e8, f4, ...]
    """
    test_df = pd.DataFrame(columns=TRAIN_COLUMNS)
    y_pred = []
    matchup_list = []
    
    # Track if input has region column
    has_region = 'region' in start_df.columns
    
    rounds = [start_df.reset_index(drop=True)]
    
    for round_idx in range(n_rounds):
        cur = rounds[-1]
        winners = []
        winner_regions = []
        
        for i in range(len(cur) // 2):
            h = cur.iloc[i]
            l = cur.iloc[-i-1]
            winner, holder, matchup, pred = play_one_match(h, l, round_idx, models, sim_type)
            winners.append(winner)
            if has_region:
                winner_regions.append(h['region'])  # Winner gets the region of the higher seed
            test_df = pd.concat([test_df, holder], ignore_index=True)
            y_pred.append(pred)
            matchup_list.append(matchup)
            print(f"{h['SEED']} {h['TEAM']}  vs.  {l['SEED']} {l['TEAM']}")
            print("Winner:", winner['SEED'], winner['TEAM'])
        
        winners_df = pd.DataFrame(winners, columns=DF_HEADERS)
        if has_region:
            winners_df['region'] = winner_regions
        rounds.append(winners_df)
        print("\n")
    
    return rounds


# ==================== Main Simulation ====================

def run_full_year_simulation(year, sim_type, model_combo):
    """
    Run a complete tournament simulation for a year with a specific model combination.
    
    Args:
        year: year to simulate
        sim_type: 'master', 'weeks', or 'seed_diff'
        model_combo: tuple of model names for this combo
    
    Returns:
        sim_json: dict with round results
    """
    # Load models for this combo
    models = load_models_for_combo(sim_type, model_combo)
    
    # Load master data
    master_df = pd.read_csv("master_ncaa.csv")
    
    # Get regional data
    test_regions = []
    for region in REGIONS:
        test_regions.append(
            master_df[(master_df.REGION == region) & (master_df.YEAR == year)].sort_values('SEED')
        )
    
    # Initialize result dataframes
    r64_r = pd.DataFrame(columns=DF_HEADERS)
    r32_r = pd.DataFrame(columns=DF_HEADERS)
    s16_r = pd.DataFrame(columns=DF_HEADERS)
    e8_r = pd.DataFrame(columns=DF_HEADERS)
    f4_r = pd.DataFrame(columns=DF_HEADERS)
    c2_r = pd.DataFrame(columns=DF_HEADERS)
    winner_r = pd.DataFrame(columns=DF_HEADERS)
    
    # Process each region
    for region_idx, region_df in enumerate(test_regions):
        r64_r = pd.concat([r64_r, region_df], ignore_index=True)
        rounds = run_rounds(region_df, n_rounds=4, models=models, sim_type=sim_type)
        r32_r = pd.concat([r32_r, rounds[1]], ignore_index=True)
        s16_r = pd.concat([s16_r, rounds[2]], ignore_index=True)
        e8_r = pd.concat([e8_r, rounds[3]], ignore_index=True)
        
        # Add region to F4 winners before concatenating
        f4_winners = rounds[4].copy()
        f4_winners['region'] = REGIONS[region_idx]
        f4_r = pd.concat([f4_r, f4_winners], ignore_index=True)
        
        print("_" * 40)
    
    # Final rounds (F4 and Championship)
    final_rounds = run_rounds(f4_r, n_rounds=2, models=models, sim_type=sim_type)
    c2_r = pd.concat([c2_r, final_rounds[1]], ignore_index=True)
    winner_r = pd.concat([winner_r, final_rounds[2]], ignore_index=True)
    
    # Build output JSON
    sim_json = {}
    for df in [r64_r, r32_r, s16_r, e8_r, f4_r, c2_r, winner_r]:
        # Only assign regions if not already present (e.g., f4_r already has regions)
        if 'region' not in df.columns:
            if len(df) > 2:
                repeats = int(len(df) / 4)
                result = np.repeat(REGIONS, repeats).tolist()
                df['region'] = result
            else:
                df['region'] = None
        sim_json.update({str(len(df)): df[['SEED', 'TEAM', 'region']].to_dict(orient="records")})
    
    return sim_json


def get_combo_filename(sim_type, model_combo):
    """Generate a descriptive filename for a model combo."""
    if sim_type == "master":
        return f"master_{model_combo[0]}"
    elif sim_type == "weeks":
        w1, w2, ff = model_combo
        return f"weeks_w1_{w1}_w2_{w2}_ff_{ff}"
    elif sim_type == "seed_diff":
        big, little, comp = model_combo
        return f"seed_diff_big_{big}_little_{little}_comp_{comp}"


# ==================== Main Entry Point ====================

def main():
    parser = argparse.ArgumentParser(description='Run all model permutations for NCAA tournament simulation.')
    parser.add_argument('--year', type=int, help='Year to process', default=current_year)
    parser.add_argument('--sim_type', type=str, help='Specific sim type to run', 
                       choices=['master', 'weeks', 'seed_diff'], default=None)
    parser.add_argument('--dry_run', action='store_true', help='Print permutations without running')
    parser.add_argument('--sample_n', type=int, help='Run only first N permutations per sim_type', default=None)
    args = parser.parse_args()
    
    # Determine which sim_types to run
    sim_types_to_run = [args.sim_type] if args.sim_type else ['master', 'weeks', 'seed_diff']
    
    # Count total permutations
    total_count = 0
    all_perms = {}
    
    for sim_type in sim_types_to_run:
        perms_list = list(get_permutations(sim_type))
        all_perms[sim_type] = perms_list
        total_count += len(perms_list)
    
    print(f"Total permutations to run: {total_count}")
    if args.dry_run:
        for sim_type in sim_types_to_run:
            print(f"\n{sim_type.upper()} ({len(all_perms[sim_type])} combos):")
            for i, combo in enumerate(all_perms[sim_type][:5]):  # Show first 5
                print(f"  {i+1}. {get_combo_filename(sim_type, combo)}")
            if len(all_perms[sim_type]) > 5:
                print(f"  ... and {len(all_perms[sim_type]) - 5} more")
        return
    
    # Create output directory if needed
    output_dir = Path(f"./Sims/{args.year}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run simulations
    combo_count = 0
    for sim_type in sim_types_to_run:
        print(f"\n{'='*60}")
        print(f"Running {sim_type.upper()} simulations (Year {args.year})")
        print(f"{'='*60}\n")
        
        for perm_idx, model_combo in enumerate(all_perms[sim_type]):
            combo_count += 1
            
            # Check if we should stop based on sample_n
            if args.sample_n and perm_idx >= args.sample_n:
                print(f"\nStopped after {args.sample_n} permutations (--sample_n limit reached)")
                break
            
            combo_name = get_combo_filename(sim_type, model_combo)
            print(f"[{combo_count}/{total_count}] Running {combo_name}...")
            
            try:
                sim_json = run_full_year_simulation(args.year, sim_type, model_combo)
                
                # Save to file
                timestamp = int(datetime.now().timestamp())
                filename = f"{combo_name}_{timestamp}.json"
                filepath = output_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump(sim_json, f, indent=4)
                
                print(f"✓ Saved to {filepath}\n")
            
            except Exception as e:
                print(f"✗ Error running {combo_name}: {str(e)}\n")
                continue
    
    print(f"\n{'='*60}")
    print(f"Batch simulation complete! Ran {combo_count} permutations.")
    print(f"Results saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
