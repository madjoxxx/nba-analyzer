import numpy as np
import pandas as pd

from nba_api.stats.endpoints import playergamelog
from sklearn.linear_model import LinearRegression


# -------------------------
# LOAD GAMES — SAFE
# -------------------------

def get_games(pid):

    try:
        df = playergamelog.PlayerGameLog(player_id=pid).get_data_frames()[0]
        df = df.sort_values("GAME_DATE")

        if len(df) < 15:
            return None

        return df

    except:
        return None


# -------------------------
# OPPONENT
# -------------------------

def get_opponent(row):
    try:
        return row["MATCHUP"].split()[-1]
    except:
        return "UNK"


# -------------------------
# OPPONENT ADJUSTMENT
# -------------------------

def opponent_adjustment(df):

    try:
        df["OPP"] = df.apply(get_opponent, axis=1)
        season_avg = df["PTS"].mean()

        tab = df.groupby("OPP")["PTS"].mean()
        recent = df.tail(5)["OPP"]

        adj = 0
        n = 0

        for o in recent:
            if o in tab:
                adj += tab[o] - season_avg
                n += 1

        if n == 0:
            return 0

        return adj / n

    except:
        return 0


# -------------------------
# FEATURES
# -------------------------

def build_features(df):

    f = {}

    f["pts_l5"] = df["PTS"].tail(5).mean()
    f["pts_l10"] = df["PTS"].tail(10).mean()
    f["pts_season"] = df["PTS"].mean()

    f["min_l5"] = df["MIN"].tail(5).mean()
    f["fga_l5"] = df["FGA"].tail(5).mean()
    f["fta_l5"] = df["FTA"].tail(5).mean()

    f["form_delta"] = f["pts_l5"] - f["pts_season"]

    std = df["PTS"].tail(10).std()
    if pd.isna(std):
        std = 6

    f["std_pts"] = std

    f["opp_adj"] = opponent_adjustment(df)

    return f


# -------------------------
# ML PROJECTION
# -------------------------

def project(df):

    X = df[["MIN","FGA","FTA"]].tail(20)
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

    pred += feats["form_delta"] * 0.5
    pred += feats["opp_adj"] * 0.7

    return float(pred), feats


# -------------------------
# PROBABILITY — SAFE
# -------------------------

def prob_over(pred, line, feats):

    sigma = feats.get("std_pts", 6)

    if sigma <= 0 or np.isnan(sigma):
        sigma = 6

    z = (pred - line) / sigma

    return float(1/(1+np.exp(-z)))


# -------------------------
# BACKTEST — SAFE
# -------------------------

def backtest(df, line):

    w = df.tail(12)

    if len(w) == 0:
        return 0.5

    return float((w["PTS"] > line).mean())


# -------------------------
# CONFIDENCE
# -------------------------

def confidence(p, hit):

    return round(p*60 + hit*40, 1)


# -------------------------
# VOL / CONS — SAFE
# -------------------------

def volatility(df):

    s = df["PTS"].tail(10).std()

    if pd.isna(s):
        return "MED"

    if s < 5: return "LOW"
    if s < 9: return "MED"
    return "HIGH"


def consistency(df):

    m = df["PTS"].mean()
    s = df["PTS"].std()

    if m == 0 or pd.isna(s):
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
