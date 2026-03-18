# %%
import pandas as pd
import json

# %%
master_df = pd.read_csv("master_ncaa.csv")

# %%
train_years = [13,14,15,16,17,18,19,21,22,23,24,25]
regions = ['east','south','midwest','west']

# %%
df_past64 = []
for year in train_years:
    for region in regions:
        df_past64.append(master_df[(master_df.REGION == region) & (master_df.YEAR ==2000+year)].sort_values('SEED'))
        print(year, region)

# %%
train_columns = [
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
 "POWER"
 ]

# %%
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

# %%
def get_target_variable(df, nxt_round_num, rnd_name):
    """
    This function used to get the target variable attributes from the correct column
    The list is reversed because we want to see if the lower seed wins

    df: dataframe used
    nxt_round_num: number of teams in the next round
    rnd_name: the column name of the dummy round variable of the next round

    output: list of target variable attributes (will be its own column)
    """
    
    listB = list(df[rnd_name])
    listB.reverse()
    listB = listB[0:nxt_round_num]
    return listB

# %%
def create_training_record(df, matchup_num, reseed = True):
    """
    This function takes the dataframe, and creates a new dataset of differences that can be trained
    
    df: dataframe of the round being processed
    matchup_num: the number of matchups
    """
    
    listDF = []
    for y in range(0,matchup_num):
        test_upsetH = df.iloc[y]
        test_upsetL = df.iloc[-(y+1)]
        listDF.append(get_upset_differences(test_upsetH,test_upsetL))
    if reseed:
        return pd.DataFrame(listDF, columns = train_columns).sort_values(by = ["SEED"])
    else:
        return pd.DataFrame(listDF, columns = train_columns).sort_values(by = ["SEED"])

# %%
def creation(df, next_rounds):
    """
    This function takes the current round's dataframe and the next round string name to use the previous functions to build the training df

    df: current round's dataframe
    next_rounds
    """
    
    y = pd.DataFrame(columns = train_columns)
    asd = []
    nxt_round_num = int(len(df)/2)
    upset= get_target_variable(df,nxt_round_num, next_rounds)
    for x in range(0,nxt_round_num):
        h = df.iloc[x]
        l = df.iloc[(-x-1)]
        if upset[x] == 1:
            asd.append(l)
        if upset[x] == 0:
            asd.append(h)
    next_df = pd.DataFrame(asd)
    y = pd.concat([y, create_training_record(df,nxt_round_num)], ignore_index=True)
    y["TRAIN"] = upset
    y.TRAIN = y.TRAIN.astype(int)
    return y, next_df

# %%
def create_train(a, next_round):
    df = pd.DataFrame(columns = train_columns)
    df_next = []
    for x in a:
        train, next_df = creation(x,next_round)
        df = pd.concat([df,train], ignore_index = True)
        df_next.append(next_df)
    return df, df_next

# %%
round_name = ["R32", "S16", "E8", "F4"]
train_master = pd.DataFrame(columns = train_columns)
train_w1 = pd.DataFrame(columns = train_columns)
train_w2 = pd.DataFrame(columns = train_columns)

for x in range(0,len(round_name)):
    a = 2**(6-x)
    b = 2**(5-x)
    holder = create_train(globals()["df_past" + '%s' % a], round_name[x])
    globals()["train" + '%s' % a] = holder[0]
    globals()["df_past" + '%s' % b] = holder[1]
    train_master = pd.concat([train_master,holder[0]], ignore_index = True)
    print("end")
    if x <= 1:
        train_w1 = pd.concat([train_w1,holder[0]], ignore_index = True)
    else:
        train_w2 = pd.concat([train_w2,holder[0]], ignore_index = True)



# %%
with open("data/final_four.json", "r") as file:
    f4 = json.load(file)

# %%
df_past4 = []
for year in train_years:
        order = f4[str(year+2000)]
        df = master_df[ (master_df.YEAR ==2000+year) & (master_df.games_won >= 5)].sort_values('SEED')
        df = df.set_index('TEAM').reindex(order).reset_index()

        df_past4.append(df)


# %%
train4, df_past2 = create_train(df_past4, "C2")
train2, df_past1 = create_train(df_past2, "Champions")

# %%
train_ff = pd.DataFrame(columns = train_columns)

for x in [train4,train2]:
    train_master = pd.concat([train_master,x], ignore_index = True)
    train_ff = pd.concat([train_ff, x], ignore_index = True)

# %%
def reset_positive_seeds(df):
    train_neg = df[df["SEED"]<=0]
    train_pos = df[df["SEED"]>0]
    train_pos = - train_pos
    train_pos["TRAIN"] = train_pos["TRAIN"] + 1
    return pd.concat([train_pos, train_neg]).reset_index(drop = True)


# %%
train_master= reset_positive_seeds(train_master)
train_ff = reset_positive_seeds(train_ff)
train_w1 = reset_positive_seeds(train_w1)
train_w2 = reset_positive_seeds(train_w2)

# %%
seed_cutoff_low = -7
seed_cutoff_high = -4
big_upset = train_master[train_master["SEED"] < seed_cutoff_low]
little_upset = train_master[train_master["SEED"] > seed_cutoff_high]
competative = train_master[(train_master["SEED"] <= seed_cutoff_high) & (train_master["SEED"] >= seed_cutoff_low)]

# %%
for i, x in [(train_master, 'master'), (train_ff, 'ff'), (train_w1, 'w1'), (train_w2, 'w2'), (big_upset, 'big'), (little_upset, 'little'), (competative, 'comp')]:
    i.to_csv("data/training/" + x + ".csv", index = False)


# %%
