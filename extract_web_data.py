from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import argparse
import sys
import time

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--year', type=int, help='Year to process')
args = parser.parse_args()

driver = webdriver.Chrome()

year = args.year
if year:
    print(f"Loading data for year {year}...")
    driver.get(f"https://barttorvik.com/#")
    
    # Wait for the table to be present (max 15 seconds)
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )
        print("Table loaded. Parsing...")
        time.sleep(2)  # Extra buffer to ensure all content is rendered
    except Exception as e:
        driver.quit()
        sys.exit(f"Timeout waiting for table to load: {e}")
    
    try:
        tables = pd.read_html(driver.page_source)
    except ValueError as e:
        driver.quit()
        sys.exit(f"No tables found on the page: {e}")
else:
    driver.quit()
    sys.exit("Please provide a year to process using --year argument.")

df = tables[0]

df.columns = [col[1] for col in df.columns]

# Debug: Check what the Team column looks like
print("Sample Team values:")
print(df['Team'].head(10))

# More robust splitting - split team name from ranking
# The format is typically "Rank Team Name" or similar
df[['TEAM', 'result']] = df['Team'].str.split(r'\s+', n=1, expand=True)
df['TEAM'] = df['TEAM'].str.extract(r'([A-Za-z\s&\'-]+)', expand=False).str.strip()

# If the split didn't work as expected, try alternative approach
if df['TEAM'].isna().any() or df['result'].isna().any():
    print("Warning: Some team names could not be split properly. Using alternative method...")
    # Alternative: just use the entire Team column as TEAM name
    df['TEAM'] = df['Team'].str.replace(r'^\d+\s*', '', regex=True).str.strip()

# Handle Rec column split - check if it exists and split accordingly
if 'Rec' in df.columns:
    rec_split = df['Rec'].str.split('-', n=1, expand=True)
    if rec_split.shape[1] >= 2:
        df['W'] = pd.to_numeric(rec_split[0], errors='coerce')
        df['L'] = pd.to_numeric(rec_split[1], errors='coerce')
    else:
        print("Warning: Could not split Rec column properly")
        df['W'] = None
        df['L'] = None
else:
    print("Warning: 'Rec' column not found")
    df['W'] = None
    df['L'] = None
df = df[df.Rk!='Rk']

for z in ['AdjOE', 'AdjDE', 'Barthag', 'EFG%',
       'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P%', '2P%D',
       '3P%', '3P%D', '3PR', '3PRD', 'Adj T.', 'WAB']:
    df[f'{z}1'] = pd.to_numeric(df[z].str.split(' ').str[0], errors='coerce')

df['POSTSEASON'] = None
df['SEED'] = None
df['YEAR'] = year  # Use the year from arguments
df['G'] = pd.to_numeric(df["G"], errors='coerce')
df['W'] = pd.to_numeric(df["W"], errors='coerce')



df_new = df[[
       'TEAM', 'Conf', 'G', 'W', 'AdjOE1',
         'AdjDE1', 'Barthag1', 'EFG%1', 'EFGD%1',
       'TOR1', 'TORD1', 'ORB1', 'DRB1', 
       'FTR1', 'FTRD1', '2P%1', '2P%D1',
       '3P%1', '3P%D1', 'Adj T.1', 'WAB1', 'POSTSEASON', "SEED", 'YEAR']]

df_new.columns = [
    'TEAM', 'CONF', 'G', 'W', 'ADJOE',
      'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
       'TOR', 'TORD', 'ORB', 'DRB', 
       'FTR', 'FTRD', '2P_O', '2P_D', 
       '3P_O', '3P_D', 'ADJ_T', 'WAB', 'POSTSEASON', 'SEED', 'YEAR']

df_new.to_csv(f"data/full/cbb{year}.csv", index = False)

driver.quit()

print(f"Data extraction complete for {year}")