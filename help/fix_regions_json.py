import json
import re
from pathlib import Path

regions_file = "data/regions.json"

# Read the raw file
with open(regions_file, 'r') as f:
    content = f.read()

print("Fixing regions.json syntax errors...")

# Fix patterns like: "Team1"/ "Team2" or "Team1" / "Team2"
# Replace with proper comma-separated strings
content = re.sub(r'"\s*\/\s*"', '" / "', content)  # Normalize spacing around /

# Try to parse and validate
try:
    regions_data = json.loads(content)
    print("✓ JSON is now valid!")
except json.JSONDecodeError as e:
    print(f"✗ Still has errors: {e}")
    print(f"  Line {e.lineno}, Column {e.colno}")
    print(f"  Context: {content[max(0, e.pos-50):e.pos+50]}")
    exit(1)

# Check each year/region structure
print("\nValidating structure:")
for year, regions in regions_data.items():
    if not isinstance(regions, dict):
        print(f"✗ Year {year} is not a dict")
        continue
    
    total_teams = 0
    for region, teams in regions.items():
        if not isinstance(teams, list):
            print(f"✗ {year}/{region} is not a list")
            continue
        
        # Count teams (split by / to handle shared seeds)
        team_count = 0
        for team_entry in teams:
            if isinstance(team_entry, str):
                team_count += len([t.strip() for t in team_entry.split('/') if t.strip()])
        
        total_teams += team_count
        print(f"  {year}/{region}: {len(teams)} seeds, {team_count} teams")
    
    print(f"  {year} total: {total_teams} teams\n")

# Backup original
backup_file = regions_file + ".backup"
Path(regions_file).rename(backup_file)
print(f"✓ Backed up original to {backup_file}")

# Write fixed version
with open(regions_file, 'w') as f:
    json.dump(regions_data, f, indent=2)

print(f"✓ Fixed regions.json saved")
