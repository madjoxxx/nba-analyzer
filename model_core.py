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
# LINEUP + ROLE FEATURES
# -------------------------

def lineup_features(df):

    f = {}

    min_l5 = df["MIN"].tail(5).mean()
    min_l15 = df["MIN"].tail(15).mean()

    fga_l5 = df["FGA"].tail(5).mean()
    fga_l15 = df["FGA"].tail(15).mean()

    f["minutes_spike"] = min_l5 - min_l15
    f["usage_spike"] = fga_l5 - fga_l15

    # starter proxy
    f["starter_flag"] = 1 if min_l5 >= 30 else 0

    # role stability
    min_std = df["MIN"].tail(10).std()
    f["rotation_risk"] = min_std

    return f


# -------------------------
# FEATURE ENGINE
# -------------------------

def build_features(df):

    f = {}

    f["pts_l5"] = df["PTS"].tail(5).mean()
    f["pts_l10"] = df["PTS"].tail(10).mean()
    f["pts_season"] = df["PTS"].mean()

    f["min_l5"] = df["MIN"].tail(5).mean()
    f["min_l10"] = df["MIN"].tail(10).mean()

    f["fga_l5"] = df["FGA"].tail(5).mean()
    f["fta_l5"] = df["FTA"].tail(5).mean()

    f["shot_volume"] = f["fga_l5"] + f["fta_l5"] * 0.44

    f["form_delta"] = f["pts_l5"] - f["pts_season"]

    f["std_pts"] = df["PTS"].tail(10).std()

    lf = lineup_features(df)

    f.update(lf)

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

    # -------------------------
    # REGIME ADJUSTMENTS
    # -------------------------

    pred += feats["form_delta"] * 0.4

    pred += feats["minutes_spike"] * 0.35

    pred += feats["usage_spike"] * 0.6

    if feats["starter_flag"]:
        pred *= 1.05

    # rotation risk penalty
    if feats["rotation_risk"] > 8:
        pred *= 0.94

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

    w = df.tail(12)

    hits = (w["PTS"] > line).sum()

    return hits / len(w)


# -------------------------
# CONFIDENCE
# -------------------------

def confidence(p, hit):

    base = p*60 + hit*40

    return round(base,1)


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
    s = df["PTS"].std()

    if m == 0:
        return 50

    return round((1 - s/m)*100,1)


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
