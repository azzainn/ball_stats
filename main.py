import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

years = list(range(2011,2022))
player_url = "https://stats.nba.com/stats/leaguedashplayerstats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2021-22&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight="

# bypasses nba's bot detection
headers  = {
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

request_url = requests.get(url=player_url, headers=headers).json()
players = request_url["resultSets"][0]["rowSet"]
columns = request_url["resultSets"][0]["headers"]

stats_df = pd.DataFrame(players, columns=columns)
stats_df.to_csv("player_stats", index=False)

# # scraping player data 
# for year in years:
#     # changing url based on year
#     url = player_url.format(str(year) + "-" + str(year + 1) )

#     # requesting url and storing that as a json 
#     r = requests.get(url=url, headers=headers).json()
#     player_info = r["resultSets"]
#     # writing text of url to an html file
#     # with open(file=f"player/{year}.html", mode="w+", encoding="utf-8") as f:
#     #     f.write(r.text)