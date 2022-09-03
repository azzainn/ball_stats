import requests
import pandas as pd
import bs4 as bs

years = list(range(2002,2022))
player_url = "https://www.basketball-reference.com/leagues/NBA_{}_totals.html"

# scraping player data
for year in years:
    # changing url based on year
    url = player_url.format(year)

    # requesting url and storing that into an object
    r = requests.get(url)

    # writing text of url to a csv file
    with open(file="player.{year}", mode="w+", encoding="utf-8") as f:
        f.write(r.text)
