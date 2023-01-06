import requests
import numpy as np
import pandas as pd
import os.path
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import (
    SelectKBest, f_regression, RFE
)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import (Ridge, Lasso, LinearRegression)
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt

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
    """Retrieves general stats for all players from NBA.com for a given year.

    Args:
        year (int): year to retrieve stats from
    Returns:
        DataFrame of stats from given year

    """
    year_url = player_url.format(str(year) + "-" + str(year + 1)[2:])
    print(year) # To check if request timed out
    request_year_url = requests.get(url=year_url, headers=headers).json()

    players = request_year_url["resultSets"][0]["rowSet"]
    columns = request_year_url["resultSets"][0]["headers"]

    stats_year = pd.DataFrame(players, columns=columns)

    return stats_year


def format_stats_year(df, year):
    """Drops unnecessary features, adds a year column, turns floatable strings to floats.

    Args:
        df: DataFrame to format.
    Returns:
        None

    """
    df.drop(
        [
            "PLAYER_ID",
            "NICKNAME",
            "TEAM_ABBREVIATION",
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

    df["YEAR"] = year

    for column in df:
        try:
            df[column] = df[column].apply(lambda x: float(x))
        except:
            pass


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
        file (str): name of file to retrieve data from / output data to
    Returns:
        tuple contains DataFrame of all NBA player stats from 2001-2022 and DataFrame of player names

    """
    stats = []

    if not os.path.exists(file):

        for year in years:
            stats_year = get_stats_year(year)
            format_stats_year(stats_year, year)
            stats.append(stats_year)

        stats = pd.concat(stats)
        player_names = stats["PLAYER_NAME"]
        stats.drop("PLAYER_NAME", inplace=True, axis=1)
        stats.to_csv(file, index=False)
        player_names.to_csv("player_names.csv", index=False)

    else:
        stats = pd.read_csv(file, index_col=False)
        player_names = pd.read_csv("player_names.csv", index_col=False)
    return (stats, player_names)


def keep_best_features(df, target, k):
    """Keeps k columns with the best features from dataset using ensemble feature selection.
    
    Args:
        df: the DataFrame with all your data
        target (str): the feature you want to test correlation for
        k (int): num features to keep
    Returns:
        DataFrame containing only the selected features

    """
    X = df.drop(target, axis=1)
    y = df[target]

    # Pearson
    pearson_corr = SelectKBest(f_regression, k=k)
    pearson_corr.fit_transform(X, y)
    selected_features_pearson = X.columns[pearson_corr.get_support()]

    # RFE
    rfe = RFE(LinearRegression(), n_features_to_select=k)
    rfe.fit_transform(X, y)
    selected_features_rfe = X.columns[rfe.get_support()]

    # Ensemble
    selected_features = set(selected_features_pearson).intersection(selected_features_rfe)
    selected_stats = df[list(selected_features)]
    
    return selected_stats


def split_into_sets(df, target, train_size, valid_size, k):
    """Splits a dataset into training, validation, and testing feature & target data based on chronological order
    
    Args:
        df: DataFrame to split
        target (str): feature to train on
        train_size (float): value between 0 and 1 determining size of training set
        valid_size (float): value between 0 and 1 determining size of validation set
        k (int): num features to keep
    Returns:
        tuple containing training, validation, and testing feature & target data

    """
    if train_size + valid_size >= 1:
        raise ValueError("train_size + valid_size must be < 1")

    train_index = int(len(df)*train_size)
    valid_index = int(len(df)*(train_size+valid_size))
    selected_stats = keep_best_features(df, target, k)
    X_train, y_train = selected_stats[0:train_index], df[0:train_index][target]
    X_valid, y_valid = selected_stats[train_index:valid_index], df[train_index:valid_index][target]
    X_test, y_test = selected_stats[valid_index:], df[valid_index:][target]

    return (X_train, y_train, X_valid, y_valid, X_test, y_test)


def best_alpha(sets): # find alpha for Ridge/Lasso using grid search
    """ Determine the best alpha value to use for Ridge and Lasso models
    
    Args:
        sets: tuple of training, validation, and testing feature & data
    Returns:
        tuple containing best alpha values

    """
    X_train, y_train, X_valid, y_valid, X_test, y_test = sets
    param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10]}
    
    ridge = Ridge()
    lasso = Lasso()

    grid_ridge = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error")
    grid_lasso = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error")

    grid_ridge.fit(X_train, y_train)
    grid_lasso.fit(X_train, y_train)

    best_alpha_ridge = grid_ridge.best_params_["alpha"]
    best_alpha_lasso = grid_lasso.best_params_["alpha"]

    return (best_alpha_ridge, best_alpha_lasso)


def best_model(models, sets, alphas):
    """ Chooses the best model from a list of models based on lowest mean squared error.

    Args:
        models (list): models to choose from
        sets (list): train, valid, and test feature & target data
        alphas (tuple): alpha values to use
    Returns:
        Model that leads to lowest mse

    """
    X_train, y_train, X_valid, y_valid, X_test, y_test = sets
    min_mse = float("inf")
    best_model = None

    for model in models:
        if model == Ridge:
            alpha = alphas[0]
            model = model(alpha=alpha)
        elif model == Lasso:
            alpha = alphas[1]
            model = model(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        mse = mean_squared_error(y_true=y_valid, y_pred=y_pred)

        if mse < min_mse:
            min_mse = mse
            best_model = model

    return best_model


# def test_model(model):
#     X_train, y_train, X_valid, y_valid, X_test, y_test = sets
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
#     print(mse)


if __name__ == "__main__":

    stats, player_names = get_stats("stats.csv")
    scale_data(stats)
    sets = split_into_sets(stats, "PTS", .7, .15, 12)
    models = [LinearRegression(), Ridge(), Lasso(), RandomForestRegressor()]
    alphas = best_alpha(sets)
    model = best_model(models, sets, (1, 0.001))

    # def train_all_model(stat_to_predict):
    #     stats_excluding_prediction = [
    #         stat
    #         for stat in numerical_stats
    #         if stat != stat_to_predict or stat != "NBA_FANTASY_PTS"
    #     ]

    #     # Split data into training, validation, and test sets
    #     training_data = stats[stats["YEAR"] < 2017]
    #     validation_data = stats[(stats["YEAR"] >= 2017) & (stats["YEAR"] <= 2019)]
    #     test_data = stats[stats["YEAR"] > 2019]

    #     model = Ridge(alpha=0.1)
    #     model.fit(
    #         training_data[stats_excluding_prediction], training_data[stat_to_predict]
    #     )

    #     # Evaluate model on validation set
    #     predicted_stats = model.predict(validation_data[stats_excluding_prediction])
    #     mse = mean_squared_error(validation_data[stat_to_predict], predicted_stats)
    #     print("Validation MSE:", mse)
    #     predicted_stats = pd.DataFrame(
    #         predicted_stats, columns=["predicted_stats"], index=validation_data.index
    #     )
    #     validation_combined_stats = pd.concat(
    #         [validation_data[["PLAYER_NAME", stat_to_predict]], predicted_stats], axis=1
    #     )

    #     # Add code here to select the best model based on mean squared error
    #     # Select the model with the best performance on the validation set

    #     # Evaluate final model on test set
    #     predicted_stats = model.predict(test_data[stats_excluding_prediction])
    #     mse = mean_squared_error(test_data[stat_to_predict], predicted_stats)
    #     print("Validation MSE:", mse)
    #     predicted_stats = pd.DataFrame(
    #         predicted_stats, columns=["predicted_stats"], index=test_data.index
    #     )
    #     combined_stats = pd.concat(
    #         [test_data[["PLAYER_NAME", stat_to_predict]], predicted_stats], axis=1
    #     )
    #     combined_stats = combined_stats.sort_values("predicted_stats", ascending=False)
    #     combined_stats["predicted_stats"] = combined_stats["predicted_stats"].apply(
    #         lambda x: 0 if x < 0 else x
    #     )  # in case of any negative predicted values
    #     combined_stats["accuracy"] = [
    #         combined_stats.iat[i, 2] / combined_stats.iat[i, 1]
    #         if abs(combined_stats.iat[i, 1] - combined_stats.iat[i, 2]) > 0.1
    #         else 1
    #         for i in range(0, len(predicted_stats))
    #     ]  # prevents runtimewarning with near equal numbers dividing by each other
    #     return combined_stats  # return the model instead

    # train_all_model("PTS").head(50)
