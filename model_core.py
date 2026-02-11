import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playergamelog
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# TEAM DEF + PACE TABLE (proxy)
# -----------------------------

TEAM_DEF_RANK = {
    "BOS": 4, "MIL": 7, "NYK": 9, "MIA": 6,
    "LAL": 18, "DAL": 20, "ATL": 25, "IND": 27,
    "SAS": 28, "UTA": 29, "WAS": 30
}

TEAM_PACE = {
    "IND": 103, "ATL": 102, "SAS": 101,
    "DAL": 99, "BOS": 98, "NYK": 96
}


# -----------------------------
# DATA
# -----------------------------

def get_games(pid):
    gl = playergamelog.PlayerGameLog(player_id=pid)
    df = gl.get_data_frames()[0]

    df["HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)
    df["OPP"] = df["MATCHUP"].apply(lambda x: x.split()[-1])

    df = df.sort_values("GAME_DATE")
    df.reset_index(drop=True, inplace=True)

    return df.tail(30)


# -----------------------------
# FEATURES
# -----------------------------

def add_features(df):

    df["MIN_R5"] = df["MIN"].rolling(5).mean()
    df["PTS_R5"] = df["PTS"].rolling(5).mean()
    df["PTS_R10"] = df["PTS"].rolling(10).mean()

    df["FGA_R5"] = df["FGA"].rolling(5).mean()
    df["FG3A_R5"] = df["FG3A"].rolling(5).mean()

    df["VOL"] = df["PTS"].rolling(10).std()

    df["USAGE_PROXY"] = df["FGA_R5"] + df["FG3A_R5"] * 0.5

    df["DEF_RANK"] = df["OPP"].map(TEAM_DEF_RANK).fillna(15)
    df["PACE"] = df["OPP"].map(TEAM_PACE).fillna(100)

    df.dropna(inplace=True)

    return df


# -----------------------------
# ML MODEL
# -----------------------------

def train_model(df):

    feats = [
        "MIN_R5","PTS_R5","PTS_R10",
        "FGA_R5","FG3A_R5",
        "VOL","HOME",
        "USAGE_PROXY","DEF_RANK","PACE"
    ]

    X = df[feats]
    y = df["PTS"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X, y)

    return model, feats


# -----------------------------
# PROBABILITY ENGINE
# -----------------------------

def calc_probs(pred, line, vol):

    if vol < 2:
        vol = 2

    z = (pred - line) / vol
    over = 1 / (1 + np.exp(-z))
    under = 1 - over

    return over*100, under*100


# -----------------------------
# ELITE PICK CLASSIFIER — FIXED
# -----------------------------

def classify_pick(pred, line, over_p, under_p, edge):

    # ELITE OVER
    if over_p >= 78 and edge >= 6 and pred >= line + 3:
        return "ELITE_OVER"

    # ELITE UNDER — FIXED LOGIC
    if under_p >= 80 and edge >= 7 and pred <= line - 4:
        return "ELITE_UNDER"

    if over_p >= 60:
        return "OVER"

    if under_p >= 60:
        return "UNDER"

    return "PASS"


# -----------------------------
# MAIN RUN
# -----------------------------

def run_model(pid, line, opp_override=None):

    df = get_games(pid)
    df = add_features(df)

    if len(df) < 12:
        return None

    if opp_override:
        df.loc[df.index[-1], "OPP"] = opp_override

    model, feats = train_model(df)

    last = df.iloc[-1][feats]
    pred = model.predict([last])[0]

    vol = df["VOL"].iloc[-1]

    over_p, under_p = calc_probs(pred, line, vol)

    edge = abs(pred - line)

    pick = classify_pick(pred, line, over_p, under_p, edge)

    conf = min(0.95, (100-vol)/100)

    return {
        "Pred": round(pred,1),
        "Line": line,
        "P_over_%": round(over_p,1),
        "P_under_%": round(under_p,1),
        "Edge": round(edge,1),
        "Confidence": round(conf,2),
        "PickType": pick
    }
