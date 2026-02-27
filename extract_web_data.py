from selenium import webdriver
import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--year', type=int, help='Year to process')
args = parser.parse_args()

driver = webdriver.Chrome()

year = args.year
if year:
    driver.get(f"https://barttorvik.com/trank.php?year=20{year}#")
    # Wait for page to load
    tables = pd.read_html(driver.page_source)
else:
    sys.exit("Please provide a year to process using --year argument.")

df = tables[0]

df.columns = [col[1] for col in df.columns]
df[['TEAM', 'result']] = df['Team'].str.split(r'(?=\d)', n=1, expand=True)
df[['W', 'L']] = df['Rec'].str.split('-', n=1, expand=True)
df = df[df.Rk!='Rk']

for z in ['AdjOE', 'AdjDE', 'Barthag', 'EFG%',
       'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P%', '2P%D',
       '3P%', '3P%D', '3PR', '3PRD', 'Adj T.', 'WAB']:
    df[f'{z}1'] = pd.to_numeric(df[z].str.split(' ').str[0])

df['POSTSEASON'] = None
df['SEED'] = None
df['YEAR'] = 2022
df['G'] = pd.to_numeric(df["G"])
df['W'] = pd.to_numeric(df["W"])



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

print(f"Data extraction complete for {year}")