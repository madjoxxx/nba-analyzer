import numpy as np
import pandas as pd

from nba_api.stats.endpoints import playergamelog
from sklearn.linear_model import LinearRegression


# -------------------------
# LOAD GAMES
# -------------------------

def get_games(pid):

    try:
        df = playergamelog.PlayerGameLog(
            player_id=pid,
            season="2024-25"
        ).get_data_frames()[0]

        df = df.sort_values("GAME_DATE")

        if len(df) < 15:
            return None

        return df

    except:
        return None


# -------------------------
# FEATURE ENGINE
# -------------------------

def build_features(df):

    f = {}

    f["pts_l5"] = df["PTS"].tail(5).mean()
    f["pts_l10"] = df["PTS"].tail(10).mean()
    f["pts_season"] = df["PTS"].mean()

    # minutes model
    f["min_l5"] = df["MIN"].tail(5).mean()
    f["min_l10"] = df["MIN"].tail(10).mean()
    f["min_trend"] = f["min_l5"] - f["min_l10"]

    # usage proxy
    f["fga_l5"] = df["FGA"].tail(5).mean()
    f["fta_l5"] = df["FTA"].tail(5).mean()

    # shot volume score
    f["shot_volume"] = f["fga_l5"] + f["fta_l5"] * 0.44

    # efficiency
    f["fg_pct"] = df["FG_PCT"].tail(10).mean()

    # form regime
    f["form_delta"] = f["pts_l5"] - f["pts_season"]

    # volatility
    f["std_pts"] = df["PTS"].tail(10).std()

    return f


# -------------------------
# ML PROJECTION
# -------------------------

def project(df):

    X = df[[
        "MIN","FGA","FTA"
    ]].tail(20)

    y = df["PTS"].tail(20)

    if len(X) < 10:
        return None, None

    model = LinearRegression()
    model.fit(X, y)

    feats = build_features(df)

    pred = model.predict([[
        feats["min_l5"],
        feats["fga_l5"],
        feats["fta_l5"]
    ]])[0]

    # regime adjustment
    pred += feats["form_delta"] * 0.4

    # minutes trend adjustment
    pred += feats["min_trend"] * 0.3

    # shot volume adjustment
    pred += (feats["shot_volume"] - X["FGA"].mean()) * 0.2

    return float(pred), feats


# -------------------------
# PROBABILITY
# -------------------------

def prob_over(pred, line, feats):

    sigma = max(feats["std_pts"], 4)

    z = (pred - line) / sigma

    p = 1 / (1 + np.exp(-z))

    return float(p)


# -------------------------
# BACKTEST SAFE
# -------------------------

def backtest(df, line):

    if len(df) < 12:
        return 0.5

    window = df.tail(12)

    hits = (window["PTS"] > line).sum()

    return hits / len(window)


# -------------------------
# CONFIDENCE
# -------------------------

def confidence(p, hit):

    return round(p*60 + hit*40, 1)


# -------------------------
# VOLATILITY
# -------------------------

def volatility(df):

    s = df["PTS"].tail(10).std()

    if s < 5:
        return "LOW"
    if s < 9:
        return "MED"
    return "HIGH"


def consistency(df):

    m = df["PTS"].mean()
    std = df["PTS"].std()

    if std == 0:
        return 50

    return round((1 - std/m) * 100,1)


# -------------------------
# STAKE
# -------------------------

def stake(edge, conf):

    if conf > 70 and abs(edge) > 3:
        return 5
    if conf > 65:
        return 4
    if conf > 58:
        return 3
    return 2
