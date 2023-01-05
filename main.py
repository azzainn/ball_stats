import math
from math import comb
from operator import ge
from matplotlib import pyplot as plt
import requests
import numpy as np
import pandas as pd
import os.path
from sklearn.linear_model import (
    Ridge,
)  # useful form of linear regression to prevent overfitting
from sklearn.metrics import mean_squared_error  # useful error metric for regression
from sklearn.preprocessing import MinMaxScaler

years = list(range(2001, 2022))
player_url = "https://stats.nba.com/stats/leaguedashplayerstats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season={}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight="

headers = {  # need this to bypass nba's bot detection
    "Connection": "keep-alive",
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-token": "true",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36",
    "x-nba-stats-origin": "stats",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Referer": "https://stats.nba.com/",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
}
# helpers
def get_stats_year(year):
    """Retrieves general stats for all players from NBA.com for a given year
    Args:
        year: year to retrieve stats from
    Returns:
        DataFrame of stats from given year
    """
    year_url = player_url.format(
        str(year) + "-" + str(year + 1)[2:]
    )
    request_year_url = requests.get(url=year_url, headers=headers).json()

    players = request_year_url["resultSets"][0][
        "rowSet"
    ]
    columns = request_year_url["resultSets"][0]["headers"]

    stats_year = pd.DataFrame(players, columns=columns)

    return stats_year


def clean_stats_year(df, year):
    """Drops unnecessary features and adds a year column.
    Args:
        df: DataFrame to clean up.
    Returns:
        None
    """
    df.drop(
        [
            "PLAYER_ID",
            "TEAM_ID",
            "WNBA_FANTASY_PTS",
            "WNBA_FANTASY_PTS_RANK",
            "CFID",
            "CFPARAMS",
            "GP_RANK",
            "W_RANK",
            "L_RANK",
            "W_PCT_RANK",
            "MIN_RANK",
            "FGM_RANK",
            "FGA_RANK",
            "FG_PCT_RANK",
            "FG3M_RANK",
            "FG3A_RANK",
            "FG3_PCT_RANK",
            "FTM_RANK",
            "FTA_RANK",
            "FT_PCT_RANK",
            "OREB_RANK",
            "DREB_RANK",
            "REB_RANK",
            "AST_RANK",
            "TOV_RANK",
            "STL_RANK",
            "BLK_RANK",
            "BLKA_RANK",
            "PF_RANK",
            "PFD_RANK",
            "PTS_RANK",
            "PLUS_MINUS_RANK",
            "NBA_FANTASY_PTS_RANK",
            "DD2_RANK",
            "TD3_RANK",
        ],
        axis=1,
        inplace=True,
    )
    df["YEAR"] = str(year)


def scale_data(df):
    """Scales a DataFrame using MinMaxScaler.
    Args:
        df: DataFrame to scale
    Returns:
        None
    """
    numeric_cols = df.columns[df.dtypes==np.number]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols])
    scaled_data = pd.DataFrame(scaled_data, columns=numeric_cols)
    df = df.drop(columns=numeric_cols)
    df = df.join(scaled_data)


def get_stats(file):
    """ Get NBA player stats from 2001-2022.
    Args:
        file: name of file to retrieve data from / output data to
    Returns:
        pandas DataFrame of all NBA player stats from 2001-2022
    """
    stats = []

    if not os.path.exists(file):

        for year in years:
            stats_year = get_stats_year(year)
            clean_stats_year(stats_year, year)
            stats_year["YEAR"] = f"{year}"
            stats.append(stats_year)
        stats = pd.concat(stats)
        stats.to_csv(file, index=False)

    else:
        stats = pd.read_csv(file)
    
    return stats

if __name__ == "__main__":

    stats = get_stats("stats.csv")
    scale_data(stats)

    def train_all_model(stat_to_predict):
        stats_excluding_prediction = [
            stat
            for stat in numerical_stats
            if stat != stat_to_predict or stat != "NBA_FANTASY_PTS"
        ]

        # Split data into training, validation, and test sets
        training_data = stats[stats["YEAR"] < 2017]
        validation_data = stats[(stats["YEAR"] >= 2017) & (stats["YEAR"] <= 2019)]
        test_data = stats[stats["YEAR"] > 2019]

        model = Ridge(alpha=0.1)
        model.fit(
            training_data[stats_excluding_prediction], training_data[stat_to_predict]
        )

        # Evaluate model on validation set
        predicted_stats = model.predict(validation_data[stats_excluding_prediction])
        mse = mean_squared_error(validation_data[stat_to_predict], predicted_stats)
        print("Validation MSE:", mse)
        predicted_stats = pd.DataFrame(
            predicted_stats, columns=["predicted_stats"], index=validation_data.index
        )
        validation_combined_stats = pd.concat(
            [validation_data[["PLAYER_NAME", stat_to_predict]], predicted_stats], axis=1
        )

        # Add code here to select the best model based on mean squared error
        # Select the model with the best performance on the validation set

        # Evaluate final model on test set
        predicted_stats = model.predict(test_data[stats_excluding_prediction])
        mse = mean_squared_error(test_data[stat_to_predict], predicted_stats)
        print("Validation MSE:", mse)
        predicted_stats = pd.DataFrame(
            predicted_stats, columns=["predicted_stats"], index=test_data.index
        )
        combined_stats = pd.concat(
            [test_data[["PLAYER_NAME", stat_to_predict]], predicted_stats], axis=1
        )
        combined_stats = combined_stats.sort_values("predicted_stats", ascending=False)
        combined_stats["predicted_stats"] = combined_stats["predicted_stats"].apply(
            lambda x: 0 if x < 0 else x
        )  # in case of any negative predicted values
        combined_stats["accuracy"] = [
            combined_stats.iat[i, 2] / combined_stats.iat[i, 1]
            if abs(combined_stats.iat[i, 1] - combined_stats.iat[i, 2]) > 0.1
            else 1
            for i in range(0, len(predicted_stats))
        ]  # prevents runtimewarning with near equal numbers dividing by each other
        return combined_stats  # return the model instead

    # train_all_model("PTS").head(50)
