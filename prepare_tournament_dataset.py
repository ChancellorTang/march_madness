import pandas as pd
import json
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description='Merge tournament teams from regions.json with their stats from cbbXX.csv')
parser.add_argument('--year', type=int, help='Tournament year to process', default=26)
parser.add_argument('--output', type=str, help='Output filename', default=None)
args = parser.parse_args()

year = args.year
cbb_file = f"data/full/cbb{year}.csv"
regions_file = "data/regions.json"

# Check if files exist
if not Path(cbb_file).exists():
    sys.exit(f"File not found: {cbb_file}")
if not Path(regions_file).exists():
    sys.exit(f"File not found: {regions_file}")

print(f"Loading tournament data for {year}...")

# Load CBB stats
df_stats = pd.read_csv(cbb_file)

# Load regions
with open(regions_file) as f:
    try:
        regions_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in {regions_file}")
        print(f"  Error at line {e.lineno}, column {e.colno}: {e.msg}")
        print(f"\n  Run: python fix_regions_json.py")
        sys.exit(1)

if str(year + 2000) not in regions_data:
    sys.exit(f"Year {year} not found in regions.json")

tournament_teams = regions_data[str(year + 2000)]

# Build a list of all tournament teams with their regions
# Handle tuple/slash-separated teams (teams that share same seed)
tournament_list = []
for region, teams in tournament_teams.items():
    for seed, team_entry in enumerate(teams, 1):
        # Handle both "/" separated strings and tuples
        if isinstance(team_entry, (tuple, list)):
            team_names = team_entry
        else:
            # Split by "/" to handle "Team1 / Team2" format (shared seeds)
            team_names = [t.strip() for t in str(team_entry).split('/')]
        
        # Add each team with the same seed
        for team_name in team_names:
            if team_name:  # Skip empty strings
                tournament_list.append({
                    'TEAM': team_name,
                    'REGION': region,
                    'SEED': seed
                })

df_tournament = pd.DataFrame(tournament_list)

# Merge with stats
# Do a case-insensitive merge
df_stats['TEAM_LOWER'] = df_stats['TEAM'].str.lower()
df_tournament['TEAM_LOWER'] = df_tournament['TEAM'].str.lower()

df_merged = pd.merge(
    df_tournament,
    df_stats,
    on='TEAM_LOWER',
    how='left'
)

# Use the tournament team name (in case there are minor spelling differences)
df_merged['TEAM'] = df_merged['TEAM_x']
df_merged = df_merged.drop(['TEAM_x', 'TEAM_y', 'TEAM_LOWER'], axis=1)

# Ensure SEED and REGION columns exist
if 'SEED_x' in df_merged.columns:
    df_merged['SEED'] = df_merged['SEED_x']
    df_merged = df_merged.drop(['SEED_x', 'SEED_y'], axis=1, errors='ignore')
if 'REGION_x' in df_merged.columns:
    df_merged['REGION'] = df_merged['REGION_x']
    df_merged = df_merged.drop(['REGION_x', 'REGION_y'], axis=1, errors='ignore')

# Reorder columns
cols_order = ['TEAM', 'SEED', 'REGION', 'CONF', 'G', 'W', 'WIN_PER',
              'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D', 'TOR', 'TORD',
              'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O', '3P_D',
              'ADJ_T', 'WAB', 'POSTSEASON', 'YEAR']

# Keep only columns that exist
cols_order = [c for c in cols_order if c in df_merged.columns]
df_merged = df_merged[cols_order]

# Sort by region and seed
region_order = {'east': 1, 'south': 2, 'west': 3, 'midwest': 4}
df_merged['region_sort'] = df_merged['REGION'].map(region_order)
df_merged = df_merged.sort_values(['region_sort', 'SEED']).drop('region_sort', axis=1)

# Check for missing teams
missing = df_merged[df_merged['CONF'].isna()]
if not missing.empty:
    print(f"\nWarning: {len(missing)} teams from regions.json were not found in {cbb_file}:")
    print(missing[['TEAM', 'SEED', 'REGION']])
    print()

# Output
output_file = args.output or f"data/test/cbb{year}.csv"
df_merged.to_csv(output_file, index=False)

print(f"✓ Tournament dataset created successfully!")
print(f"  Total teams: {len(df_merged)}")
print(f"  Teams with stats: {len(df_merged[df_merged['CONF'].notna()])}")
print(f"  Output: {output_file}")

# Show summary by region
print("\nBreakdown by region:")
for region in ['east', 'south', 'west', 'midwest']:
    count = len(df_merged[df_merged['REGION'] == region])
    print(f"  {region.capitalize()}: {count} teams")
