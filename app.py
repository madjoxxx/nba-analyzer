import streamlit as st
import pandas as pd
import numpy as np

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# -------------------------
# CONFIG
# -------------------------

MIN_GAMES_REQUIRED = 18
N_ESTIMATORS = 400


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


@st.cache_data(ttl=3600)
def load_player_log(name):

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
# MATCHUP ENGINE
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
    return PACE.get(team.upper(),1.0) * DEF.get(team.upper(),1.0)


# -------------------------
# FATIGUE
# -------------------------

def fatigue_factor(df):

    if len(df) < 2:
        return 1.0

    d0 = pd.to_datetime(df.iloc[-1]["GAME_DATE"])
    d1 = pd.to_datetime(df.iloc[-2]["GAME_DATE"])

    if (d0-d1).days <= 1:
        return 0.94

    return 1.0


# -------------------------
# FEATURE PIPELINE
# -------------------------

def build_features(df):

    df = df.copy()

    df["USG_PROXY"] = df["FGA"] + 0.44*df["FTA"] + df["TOV"]

    df["PTS_L3"] = df["PTS"].rolling(3).mean()
    df["PTS_L5"] = df["PTS"].rolling(5).mean()
    df["MIN_L5"] = df["MIN"].rolling(5).mean()
    df["FGA_L5"] = df["FGA"].rolling(5).mean()

    df["EFF"] = df["FGM"] + 0.5*df["FG3M"]

    df = df.dropna()

    FEATURES = [
        "MIN",
        "FGA",
        "FG3A",
        "FTA",
        "USG_PROXY",
        "PTS_L3",
        "PTS_L5",
        "MIN_L5",
        "FGA_L5",
        "EFF"
    ]

    return df, FEATURES


# -------------------------
# TRAIN + VALIDATE
# -------------------------

def train_model(df, features):

    split = int(len(df)*0.8)

    train = df.iloc[:split]
    valid = df.iloc[split:]

    X_train = train[features]
    y_train = train["PTS"]

    X_valid = valid[features]
    y_valid = valid["PTS"]

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=7,
        random_state=42
    )

    model.fit(X_train,y_train)

    pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid,pred)

    return model, mae


# -------------------------
# NEXT GAME VECTOR
# -------------------------

def build_next_row(df, opp):

    r = df.iloc[-1].copy()

    r["USG_PROXY"] = r["FGA"] + 0.44*r["FTA"] + r["TOV"]

    r["PTS_L3"] = df["PTS"].tail(3).mean()
    r["PTS_L5"] = df["PTS"].tail(5).mean()
    r["MIN_L5"] = df["MIN"].tail(5).mean()
    r["FGA_L5"] = df["FGA"].tail(5).mean()

    r["EFF"] = r["FGM"] + 0.5*r["FG3M"]

    factor = matchup_factor(opp)

    r["FGA"] *= factor
    r["FG3A"] *= factor
    r["FTA"] *= factor

    return r


# -------------------------
# MONTE CARLO
# -------------------------

def simulate(mu, line, mae):

    sims = 10000

    sd = max(mae, mu*0.18)

    pts = np.random.normal(mu, sd, sims)
    pts = np.clip(pts,0,90)

    prob = (pts > line).mean()

    return prob, np.percentile(pts,10), np.percentile(pts,90)


# -------------------------
# UI
# -------------------------

st.title("NBA Production ML Points Model")

name = st.text_input("Player name")
opp = st.text_input("Opponent (BOS)")
line = st.number_input("Points line", value=20.5)

if st.button("Run Model"):

    df = load_player_log(name)

    if df is None or len(df) < MIN_GAMES_REQUIRED:
        st.write("Not enough games")
        st.stop()

    df, FEATURES = build_features(df)

    model, mae = train_model(df, FEATURES)

    next_row = build_next_row(df, opp)

    X = next_row[FEATURES].values.reshape(1,-1)

    base_pred = model.predict(X)[0]

    fatigue = fatigue_factor(df)

    final_pred = base_pred * fatigue

    prob, p10, p90 = simulate(final_pred, line, mae)

    confidence = max(0, 1 - mae/15)

    st.write("Projection:", round(final_pred,1))
    st.write("Model MAE:", round(mae,2))
    st.write("Confidence:", round(confidence*100,1), "%")
    st.write("Over %:", round(prob*100,1))
    st.write("Range:", round(p10,1), "-", round(p90,1))

    if prob > 0.75:
        st.success("STRONG OVER")
    elif prob > 0.60:
        st.info("LEAN OVER")
    elif prob < 0.40:
        st.warning("LEAN UNDER")
    else:
        st.write("NO EDGE")
