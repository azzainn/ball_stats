import requests
import pandas as pd
from bs4 import BeautifulSoup

years = list(range(2011,2022))
player_url = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html"

# scraping player data 
# MOVE ONTO selenium
for year in years:
    # changing url based on year
    url = player_url.format(year)

    # requesting url and storing that into an object
    r = requests.get(url)

    # writing text of url to a csv file
    with open(file=f"player/{year}.html", mode="w+", encoding="utf-8") as f:
        f.write(r.text)

# parsing player data

dfs = []

for year in years:
    with open(file=f"player/{year}.html", mode="w+", encoding="utf-8") as f:
        page = f.read()

    soup = BeautifulSoup(page, "html.parser")
    player_table = soup.find(id="per_game_stats")
    player_df = pd.read_html(str(player_table))[0]
    player_df["Year"] = year
    dfs.append(player_df)

players = pd.concat(dfs)
players.to_csv("players.csv")
