import requests
import csv
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import os.path

years = list(range(11,22))
player_url = "https://stats.nba.com/stats/leaguedashplayerstats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=20{}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight="

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
    if not os.path.exists(f"player_stats_20{year}.csv"):    # saves time rather than writing to a csv every runtime

        year_url = player_url.format(str(year) + "-" + str(year + 1))   # need to do this weird formatting b/c url has year written as, for example, "2011-2012"
        request_year_url = requests.get(url=year_url, headers=headers).json()

        players = request_year_url["resultSets"][0]["rowSet"]   # need this to navigate through html and extract only the player data
        columns = request_year_url["resultSets"][0]["headers"]

        stats_df = pd.DataFrame(players, columns=columns)
        del stats_df["PLAYER_ID"]
        del stats_df["TEAM_ID"]
        del stats_df["WNBA_FANTASY_PTS"]
        del stats_df["WNBA_FANTASY_PTS_RANK"]
        stats_df.to_csv(f"player_stats_20{year}.csv", mode="w+", index=False)

def name_search():
    player_name = input("Whose predictions would you like to see? (First Last)").title()
    players = pd.read_csv("player_stats_2011.csv")
    print(players)

name_search()