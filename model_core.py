import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# -------------------------
# FEATURE BUILDING
# -------------------------

def build_features(df):

    df = df.copy()

    df["pts_l5"] = df["PTS"].rolling(5).mean()
    df["pts_l10"] = df["PTS"].rolling(10).mean()
    df["min_l5"] = df["MIN"].rolling(5).mean()

    df["pts_std10"] = df["PTS"].rolling(10).std()
    df["trend"] = df["pts_l5"] - df["pts_l10"]

    df["rest"] = df["GAME_DATE"].diff().dt.days.fillna(2)

    df = df.dropna()

    return df


# -------------------------
# TRAIN ML
# -------------------------

def train_model(df):

    feats = ["pts_l5", "pts_l10", "min_l5", "pts_std10", "trend", "rest"]

    if len(df) < 25:
        return None, None

    X = df[feats]
    y = df["PTS"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )

    model.fit(X, y)

    return model, feats


# -------------------------
# PREDICT NEXT GAME
# -------------------------

def predict_next(df, model, feats):

    last = df.iloc[-1]

    X = pd.DataFrame([last[feats]])

    pred = model.predict(X)[0]

    return pred


# -------------------------
# PROBABILITY ESTIMATE
# -------------------------

def prob_over(pred, line, std):

    if std < 1:
        std = 1

    z = (pred - line) / std

    p = 1 / (1 + np.exp(-z))

    return float(p)


# -------------------------
# BACKTEST HIT RATE SAFE
# -------------------------

def backtest_hit_rate(df, line):

    if len(df) < 20:
        return 0.5

    tests = 0
    hits = 0

    for i in range(15, len(df)):

        window = df.iloc[:i]

        if len(window) < 15:
            continue

        avg = window["PTS"].tail(10).mean()

        if avg > line:
            tests += 1

            if df.iloc[i]["PTS"] > line:
                hits += 1

    if tests == 0:
        return 0.5

    return hits / tests


# -------------------------
# CONFIDENCE ENGINE
# -------------------------

def confidence_score(edge, prob, hitrate, vol):

    score = (
        edge * 3
        + (prob - 0.5) * 10
        + (hitrate - 0.5) * 6
        - vol * 0.15
    )

    return float(max(0, min(100, score * 10)))


# -------------------------
# TAGGING
# -------------------------

def signal_tag(prob, edge):

    if prob > 0.72 and edge > 2:
        return "ELITE"

    if prob > 0.65:
        return "STRONG"

    if prob > 0.57:
        return "LEAN"

    return "PASS"


def stake_size(conf):

    if conf > 80:
        return 3

    if conf > 65:
        return 2

    if conf > 55:
        return 1

    return 0.5


# -------------------------
# LINE CURVE
# -------------------------

def line_curve(pred, std):

    out = []

    for l in np.arange(pred - 5, pred + 6, 1):

        p = prob_over(pred, l, std)

        out.append({
            "Line": round(l, 1),
            "OverProb": round(p, 3)
        })

    return out
