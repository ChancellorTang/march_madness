import pandas as pd
import json
import sys

try:
    with open("data/regions.json", "r") as file:
        regions = json.load(file)
except json.JSONDecodeError as e:
    print(f"✗ Invalid JSON in data/regions.json")
    print(f"  Error at line {e.lineno}, column {e.colno}: {e.msg}")
    print(f"\n  Run this to fix it:")
    print(f"  python fix_regions_json.py")
    sys.exit(1)

def power_conf(df):
    test_list = []
    for x in df.CONF:
        if x in ["B10", "B12", "SEC", "P12", "BE", "ACC"]:
            test_list.append(1)
        else:
            test_list.append(0)
    return test_list

def format_ncaa_df(df, year = None):
    df = df.fillna(0)
    df.SEED = df.SEED.astype(int) 
    df["WIN_PER"] = df["W"]/df["G"]
    df["POWER"] = power_conf(df)
    df = df.reindex(columns=new_columns)
    if year is not None:
        df["YEAR"] = year
    return df
    
def round_assign(x):
    if x['POSTSEASON'] == "R64" : return [1,0,0,0,0,0,0]
    elif x['POSTSEASON'] == "R32" : return [1,1,0,0,0,0,0]
    elif x['POSTSEASON'] == "S16" : return [1,1,1,0,0,0,0]
    elif x['POSTSEASON'] == 'E8': return [1,1,1,1,0,0,0]
    elif x['POSTSEASON'] == 'F4': return [1,1,1,1,1,0,0]
    elif x['POSTSEASON'] == '2ND' or x['POSTSEASON'] == 'C2': return [1,1,1,1,1,1,0]
    elif x['POSTSEASON'] == 'Champions': return [1,1,1,1,1,1,1]
    else: return [0,0,0,0,0,0,0]

new_columns = ['TEAM',
 'CONF',
 'POSTSEASON',
 'G',
 'W',
 "WIN_PER",
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
 "POWER",
 'YEAR']

ncaam = pd.read_csv('data/tr/cbb.csv')
ncaam21 = pd.read_csv('data/tr/cbb21.csv')
ncaam22 = pd.read_csv('data/tr/cbb22.csv')
ncaam23 = pd.read_csv('data/tr/cbb23.csv')
ncaam24 = pd.read_csv('data/tr/cbb24.csv')
ncaam25 = pd.read_csv('data/tr/cbb25.csv')
ncaam26 = pd.read_csv('data/tr/cbb26.csv')

ncaam = format_ncaa_df(ncaam)
ncaam19 = ncaam[ncaam['YEAR']==2019]
ncaam18 = ncaam[ncaam['YEAR']==2018]
ncaam17 = ncaam[ncaam['YEAR']==2017]
ncaam16 = ncaam[ncaam['YEAR']==2016]
ncaam15 = ncaam[ncaam['YEAR']==2015]
ncaam14 = ncaam[ncaam['YEAR']==2014]
ncaam13 = ncaam[ncaam['YEAR']==2013]
ncaam21 = format_ncaa_df(ncaam21, 2021)
ncaam22 = format_ncaa_df(ncaam22, 2022)
ncaam23 = format_ncaa_df(ncaam23, 2023)
ncaam24 = format_ncaa_df(ncaam24, 2024)
ncaam25 = format_ncaa_df(ncaam25, 2025)
ncaam26 = format_ncaa_df(ncaam26, 2026)

def assign_dummy(df):
    dummy_df = []
    for x in range(0,len(df)):
        dummy_df.append(round_assign(df.iloc[x]))
    dummy_df = pd.DataFrame(dummy_df)
    dummy_df = dummy_df.rename(columns={0: "R64", 1: "R32", 2: "S16",3: "E8", 4:"F4",5:"C2",6:"Champions"})
    return dummy_df

def region_df(x,y,z):
    v = pd.DataFrame(x)
    v = v.rename(columns={0: "TEAM"})
    v = v.merge(y, on = 'TEAM', how='left')
    v = v.join(assign_dummy(v))
    v['REGION'] = z
    return v

master_df = pd.DataFrame()

all_years = [13,14,15,16,17,18,19,21,22,23,24,25,26]

legs = ["east",'south', 'midwest', 'west']

for x in all_years:
    for y in legs:
        z = str( 2000 + x)
        globals()[y + '%s' % x + '%s' %"_df"] = region_df(regions[z][y],  globals()['ncaam' + '%s' % x], y)
        master_df = pd.concat([master_df,globals()[y + '%s' % x + '%s' %"_df"]], ignore_index=True)

master_df['games_won'] = master_df[['R64', 'R32', 'S16', 'E8', 'F4', 'C2', 'Champions']].sum(axis=1)

master_df.to_csv("master_ncaa.csv", index=False)