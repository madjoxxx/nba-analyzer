import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playergamelog


# -------------------------
# DATA
# -------------------------

def get_games(pid, season="2024-25"):
    df = playergamelog.PlayerGameLog(
        player_id=pid,
        season=season
    ).get_data_frames()[0]

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    return df


# -------------------------
# FEATURE ENGINEERING
# -------------------------

def build_features(df):

    df = df.copy()

    df["PTS_L5"] = df["PTS"].rolling(5).mean()
    df["PTS_L10"] = df["PTS"].rolling(10).mean()

    df["MIN_L5"] = df["MIN"].rolling(5).mean()

    df["HOME"] = df["MATCHUP"].str.contains("vs").astype(int)

    df["REST"] = df["GAME_DATE"].diff().dt.days.fillna(2)

    df["FORM_WEIGHTED"] = (
        df["PTS"].ewm(span=7).mean()
    )

    return df.dropna().reset_index(drop=True)


# -------------------------
# SIMPLE ML MODEL (no sklearn)
# -------------------------

def train_linear_model(X, y):

    X = np.c_[np.ones(len(X)), X]
    w = np.linalg.pinv(X.T @ X) @ X.T @ y
    return w


def predict_linear(w, x):
    x = np.r_[1, x]
    return float(x @ w)


# -------------------------
# PROJECTION
# -------------------------

def project_points(df):

    feats = build_features(df)

    if len(feats) < 20:
        return None

    X = feats[[
        "PTS_L5",
        "PTS_L10",
        "MIN_L5",
        "HOME",
        "REST"
    ]].values

    y = feats["PTS"].values

    w = train_linear_model(X, y)

    last = X[-1]

    pred = predict_linear(w, last)

    return pred, feats


# -------------------------
# PROBABILITY
# -------------------------

def prob_over(pred, line, df):

    std = df["PTS"].std()

    if std == 0 or np.isnan(std):
        return 0.5

    z = (pred - line) / std

    p = 1 / (1 + np.exp(-z))
    return float(p)


# -------------------------
# BACKTEST (safe)
# -------------------------

def backtest_hit_rate(df, line):

    pts = df["PTS"].to_numpy()
    n = len(pts)

    if n < 15:
        return 0

    hits = 0
    tests = 0

    for i in range(12, n):

        hist = pts[:i]

        if len(hist) < 5:
            continue

        proj = hist[-5:].mean()
        actual = pts[i]

        if actual > line:
            hits += 1

        tests += 1

    if tests == 0:
        return 0

    return hits / tests


# -------------------------
# CONFIDENCE
# -------------------------

def confidence_score(over_prob, hitrate):

    return round(
        (over_prob * 0.6 + hitrate * 0.4) * 100,
        1
    )


# -------------------------
# VOLATILITY
# -------------------------

def volatility_label(df):

    std = df["PTS"].std()

    if std < 4:
        return "LOW"
    elif std < 8:
        return "MED"
    else:
        return "HIGH"


# -------------------------
# STAKE SUGGESTION
# -------------------------

def stake_size(edge, conf):

    if edge < 1:
        return 0.5
    if conf > 70:
        return 2
    if conf > 60:
        return 1.5
    return 1
