import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playergamelog


# -------------------------
# DATA LOAD
# -------------------------

def get_games(pid, season="2024-25"):

    try:
        df = playergamelog.PlayerGameLog(
            player_id=pid,
            season=season
        ).get_data_frames()[0]
    except:
        return None

    if df is None or len(df) == 0:
        return None

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    return df


# -------------------------
# FEATURES
# -------------------------

def build_features(df):

    df = df.copy()

    df["PTS_L5"] = df["PTS"].rolling(5).mean()
    df["PTS_L10"] = df["PTS"].rolling(10).mean()

    df["MIN_L5"] = df["MIN"].rolling(5).mean()

    df["USG"] = df["FGA"] + 0.44 * df["FTA"]
    df["USG_L5"] = df["USG"].rolling(5).mean()

    df["PACE"] = df["FGA"] + df["TOV"] + 0.44 * df["FTA"]
    df["PACE_L5"] = df["PACE"].rolling(5).mean()

    df["HOME"] = df["MATCHUP"].str.contains("vs").astype(int)

    df["REST"] = df["GAME_DATE"].diff().dt.days.fillna(2)

    df["FORM"] = df["PTS"].ewm(span=7).mean()

    df = df.dropna().reset_index(drop=True)

    if len(df) < 20:
        return None

    return df


# -------------------------
# LINEAR ML
# -------------------------

def train_model(X, y):

    X = np.c_[np.ones(len(X)), X]
    w = np.linalg.pinv(X.T @ X) @ X.T @ y
    return w


def predict_model(w, x):

    x = np.r_[1, x]
    return float(x @ w)


# -------------------------
# PROJECTION
# -------------------------

def project(df):

    feats = build_features(df)

    if feats is None:
        return None, None

    cols = [
        "PTS_L5",
        "PTS_L10",
        "MIN_L5",
        "USG_L5",
        "PACE_L5",
        "HOME",
        "REST"
    ]

    X = feats[cols].values
    y = feats["PTS"].values

    w = train_model(X, y)

    pred = predict_model(w, X[-1])

    return pred, feats


# -------------------------
# PROBABILITY
# -------------------------

def prob_over(pred, line, feats):

    std = feats["PTS"].std()

    if std == 0 or np.isnan(std):
        return 0.5

    z = (pred - line) / std

    return float(1 / (1 + np.exp(-z)))


# -------------------------
# BACKTEST SAFE
# -------------------------

def backtest(df, line):

    pts = df["PTS"].to_numpy()

    if len(pts) < 15:
        return 0.5

    hits = 0
    tests = 0

    for i in range(12, len(pts)):
        proj = pts[:i][-5:].mean()
        if pts[i] > line:
            hits += 1
        tests += 1

    if tests == 0:
        return 0.5

    return hits / tests


# -------------------------
# METRICS
# -------------------------

def volatility(df):

    s = df["PTS"].std()

    if s < 4: return "LOW"
    if s < 8: return "MED"
    return "HIGH"


def consistency(df):

    m = df["PTS"].mean()
    s = df["PTS"].std()

    if m == 0:
        return 0

    return round((1 - s/m) * 100, 1)


def confidence(over, hit):

    return round((over*0.6 + hit*0.4) * 100, 1)


def stake(edge, conf):

    if conf > 72 and edge > 2:
        return 2
    if conf > 64:
        return 1.5
    if conf > 58:
        return 1
    return 0.5
