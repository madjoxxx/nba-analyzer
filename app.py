import streamlit as st
import pandas as pd
import numpy as np

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

from sklearn.ensemble import RandomForestRegressor


# -------------------------
# UTIL
# -------------------------

def parse_min(x):
    try:
        return float(x)
    except:
        if isinstance(x,str) and ":" in x:
            m,s = x.split(":")
            return float(m)+float(s)/60
    return np.nan


def get_log(name):

    pl = players.find_players_by_full_name(name)
    if not pl:
        return None

    pid = pl[0]["id"]
    df = playergamelog.PlayerGameLog(player_id=pid).get_data_frames()[0]

    df["MIN"] = df["MIN"].apply(parse_min)
    df = df.dropna(subset=["MIN"])
    df = df.sort_values("GAME_DATE")

    return df


# -------------------------
# MATCHUP / FATIGUE
# -------------------------

PACE = {
    "IND":1.05,"ATL":1.05,"WAS":1.05,
    "GSW":1.04,"LAL":1.03,
    "NYK":0.96,"MIA":0.95,"CLE":0.95
}

DEF = {
    "SAS":1.06,"CHA":1.05,"DET":1.05,
    "MIN":0.94,"BOS":0.94,"NYK":0.95
}

def matchup_factor(team):
    return PACE.get(team,1.0) * DEF.get(team,1.0)


def fatigue_factor(df):

    if len(df) < 2:
        return 1.0

    d0 = pd.to_datetime(df.iloc[-1]["GAME_DATE"])
    d1 = pd.to_datetime(df.iloc[-2]["GAME_DATE"])

    if (d0-d1).days <= 1:
        return 0.95

    return 1.0


# -------------------------
# FEATURE ENGINEERING
# -------------------------

def build_features(df):

    df = df.copy()

    df["PTS_last3"] = df["PTS"].rolling(3).mean()
    df["MIN_last3"] = df["MIN"].rolling(3).mean()
    df["FGA_last3"] = df["FGA"].rolling(3).mean()

    df["PTS_last5"] = df["PTS"].rolling(5).mean()
    df["MIN_last5"] = df["MIN"].rolling(5).mean()

    df["USG_proxy"] = df["FGA"] + 0.44*df["FTA"]

    df = df.dropna()

    features = [
        "MIN",
        "FGA",
        "FG3A",
        "FTA",
        "USG_proxy",
        "PTS_last3",
        "MIN_last3",
        "FGA_last3",
        "PTS_last5",
        "MIN_last5"
    ]

    return df, features


# -------------------------
# ML TRAIN
# -------------------------

def train_model(df, features):

    X = df[features]
    y = df["PTS"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=1
    )

    model.fit(X,y)

    return model


# -------------------------
# PREDICT NEXT GAME FEATURES
# -------------------------

def next_game_row(df, opp):

    row = df.iloc[-1].copy()

    row["PTS_last3"] = df["PTS"].tail(3).mean()
    row["MIN_last3"] = df["MIN"].tail(3).mean()
    row["FGA_last3"] = df["FGA"].tail(3).mean()

    row["PTS_last5"] = df["PTS"].tail(5).mean()
    row["MIN_last5"] = df["MIN"].tail(5).mean()

    row["USG_proxy"] = row["FGA"] + 0.44*row["FTA"]

    # matchup adjust
    factor = matchup_factor(opp)
    row["FGA"] *= factor
    row["FG3A"] *= factor
    row["FTA"] *= factor

    return row


# -------------------------
# MONTE CARLO
# -------------------------

def simulate(mu, line):

    sims = 9000
    sd = mu * 0.22

    pts = np.random.normal(mu, sd, sims)
    pts = np.clip(pts, 0, 80)

    prob = (pts > line).mean()

    return prob, np.percentile(pts,10), np.percentile(pts,90)


# -------------------------
# UI
# -------------------------

st.title("NBA ML Points Predictor")

name = st.text_input("Player name")
opp = st.text_input("Opponent code (BOS)")
line = st.number_input("Points line", value=20.5)

if st.button("Predict"):

    df = get_log(name)

    if df is None or len(df) < 12:
        st.write("Not enough data")
        st.stop()

    df, feats = build_features(df)
    model = train_model(df, feats)

    row = next_game_row(df, opp)
    X_pred = row[feats].values.reshape(1,-1)

    base_pred = model.predict(X_pred)[0]

    adj = fatigue_factor(df)
    final_pred = base_pred * adj

    prob, p10, p90 = simulate(final_pred, line)

    st.write("ML Projection:", round(final_pred,1))
    st.write("Over probability:", round(prob*100,1),"%")
    st.write("Range:", round(p10,1),"-",round(p90,1))

    if prob > 0.75:
        st.success("STRONG OVER")
    elif prob > 0.60:
        st.info("LEAN OVER")
    elif prob < 0.40:
        st.warning("LEAN UNDER")
    else:
        st.write("NO EDGE")
