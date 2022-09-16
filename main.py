import math
from math import comb
from operator import ge
from matplotlib import pyplot as plt
import requests
import numpy as np
import pandas as pd
import os.path
from sklearn.linear_model import Ridge # useful form of linear regression to prevent overfitting
from sklearn.metrics import mean_squared_error # useful error metric for regression

years = list(range(2001,2022))
player_url = "https://stats.nba.com/stats/leaguedashplayerstats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season={}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight="

headers  = {    # need this to bypass nba's bot detection
    'Connection': 'keep-alive',
    'Accept': 'application/json, text/plain, */*',
    'x-nba-stats-token': 'true',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
    'x-nba-stats-origin': 'stats',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Referer': 'https://stats.nba.com/',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
}

for year in years:
    if not os.path.exists(f"player_stats_{year}.csv"):    # saves time rather than writing to a csv every runtime

        year_url = player_url.format(str(year) + "-" + str(year + 1)[2:])   # need to do this weird formatting b/c url has year written as, for example, "2011-2012"
        request_year_url = requests.get(url=year_url, headers=headers).json()

        players = request_year_url["resultSets"][0]["rowSet"]   # need this to navigate through html and extract only the player data
        columns = request_year_url["resultSets"][0]["headers"]

        stats_df = pd.DataFrame(players, columns=columns)
        
        stats_df.drop(["PLAYER_ID", "TEAM_ID", "WNBA_FANTASY_PTS", "WNBA_FANTASY_PTS_RANK", "CFID", "CFPARAMS", 'GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK',
       'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK', 'FG3A_RANK',
       'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK', 'OREB_RANK',
       'DREB_RANK', 'REB_RANK', 'AST_RANK', 'TOV_RANK', 'STL_RANK', 'BLK_RANK',
       'BLKA_RANK', 'PF_RANK', 'PFD_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK',
       'NBA_FANTASY_PTS_RANK', 'DD2_RANK', 'TD3_RANK'], axis=1, inplace=True)
        stats_df["YEAR"] = f"{year}"

        stats_df.to_csv(f"player_stats_{year}.csv", mode="w+", index=False)

def get_player_stats(player_name):

    player_stats = []   # not initializing a df to save memory 

    for year in years:
        all_stats_year = pd.read_csv(f"player_stats_{year}.csv")
        player_stats_year = all_stats_year.loc[all_stats_year["PLAYER_NAME"].str.contains(player_name, case=False)].at[0]
        
        if not player_stats_year.empty:
            player_stats.append(player_stats_year)

    player_stats_df = pd.DataFrame(player_stats)
    return player_stats_df

def get_all_stats():

    all_stats = []

    for year in years:
        all_stats_year = pd.read_csv(f"player_stats_{year}.csv")
        all_stats.append(all_stats_year)
        
    return pd.concat(all_stats)

"""use these stats as the parameter for train_model"""
numerical_stats = ['AGE', 'GP', 'W', 'L',
       'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
       'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL',
       'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS', 'NBA_FANTASY_PTS',
       'DD2', 'TD3']

def train_player_model(stat_to_predict, player_name):

    training_data = get_player_stats(player_name)[get_player_stats(player_name)["YEAR"] < 2021]
    testing_data = get_player_stats(player_name)[get_player_stats(player_name)["YEAR"] == 2021]
    # print(training_data)
    # print(testing_data)

    model = Ridge(alpha=.1)
    model.fit(training_data[[stat for stat in numerical_stats if stat != stat_to_predict]], training_data[stat_to_predict])
    
    predicted_stats = model.predict(testing_data[[stat for stat in numerical_stats if stat != stat_to_predict]])
    predicted_stats = pd.DataFrame(predicted_stats, columns=["predicted_stats"], index=testing_data.index)
    
    combined_stats = pd.concat([testing_data[["PLAYER_NAME", stat_to_predict]], predicted_stats], axis=1)
    return combined_stats


def train_all_model(stat_to_predict):

    stats_excluding_prediction = [stat for stat in numerical_stats if stat != stat_to_predict or stat != "NBA_FANTASY_PTS"]

    training_data = get_all_stats()[get_all_stats()["YEAR"] < 2021]
    testing_data = get_all_stats()[get_all_stats()["YEAR"] == 2021]
    # print(training_data)
    # print(testing_data)

    model = Ridge(alpha=.1)
    model.fit(training_data[stats_excluding_prediction], training_data[stat_to_predict])
    
    predicted_stats = model.predict(testing_data[stats_excluding_prediction])
    predicted_stats = pd.DataFrame(predicted_stats, columns=["predicted_stats"], index=testing_data.index)

    combined_stats = pd.concat([testing_data[["PLAYER_NAME", stat_to_predict]], predicted_stats], axis=1)
    combined_stats = combined_stats.sort_values("predicted_stats", ascending=False)
    combined_stats["predicted_stats"] = combined_stats["predicted_stats"].apply(lambda x : 0 if x < 0 else x) # in case of any negative predicted values
    combined_stats["accuracy"] = [combined_stats.iat[i, 2] / combined_stats.iat[i, 1] if abs(combined_stats.iat[i, 1] - combined_stats.iat[i, 2]) > 0.1 else 1 for i in range(0, len(predicted_stats))] # prevents runtimewarning with near equal numbers dividing by each other
    return combined_stats

# pred = train_all_model("NBA_FANTASY_PTS").head(50)
# print(pred)

def backtesting(stat_to_predict):

    for year in years[3:]:
        all_predicted_stats = []
        model = Ridge(alpha=.1)
        stats_excluding_prediction = [stat for stat in numerical_stats if stat != stat_to_predict]

        training_data = get_all_stats()[get_all_stats()["YEAR"] < year]
        testing_data = get_all_stats()[get_all_stats()["YEAR"] == year]

        model.fit(training_data[stats_excluding_prediction], training_data[stat_to_predict])

        predicted_stats = model.predict(testing_data[stats_excluding_prediction])
        predicted_stats = pd.DataFrame(predicted_stats, columns=["predicted_stats"], index=testing_data.index)
        combined_stats = pd.concat([testing_data[["PLAYER_NAME", stat_to_predict]], predicted_stats], axis=1)
        combined_stats = combined_stats.sort_values("predicted_stats", ascending=False)
        combined_stats["predicted_stats"] = combined_stats["predicted_stats"].apply(lambda x : 0 if x < 0 else x)
        combined_stats["accuracy"] = [combined_stats.iat[i, 2] / combined_stats.iat[i, 1] if abs(combined_stats.iat[i, 1] - combined_stats.iat[i, 2]) > 0.1 else 1 for i in range(0, len(predicted_stats))]
        all_predicted_stats.append(combined_stats)

        return all_predicted_stats

