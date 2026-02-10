import streamlit as st
import pandas as pd
import numpy as np

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

# -------------------------
# OPTIONAL ML IMPORT
# -------------------------

ML_AVAILABLE = True
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
except:
    ML_AVAILABLE = False


# -------------------------
# CONFIG
# -------------------------

MIN_GAMES_REQUIRED = 18


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

PACE = {"IND":1.05,"ATL":1.05,"WAS":1.05,"GSW":1.04,"LAL":1.03}
DEF  = {"SAS":1.06,"CHA":1.05,"DET":1.05,"MIN":0.94,"BOS":0.94}

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

    return 0.94 if (d0-d1).days <= 1 else 1.0


# -------------------------
# FORMULA MODEL
# -------------------------

def formula_projection(df, opp):

    mins = df["MIN"].tail(8)
    pts = df["PTS"].tail(8)

    mu_min = np.average(mins, weights=np.linspace(1.4,0.7,len(mins)))
    ppm = np.average(pts/mins, weights=np.linspace(1.5,0.7,len(mins)))

    proj = mu_min * ppm

    proj *= matchup_factor(opp)
    proj *= fatigue_factor(df)

    return proj


# -------------------------
# FEATURE PIPELINE (ML)
# -------------------------

def build_features(df):

    df = df.copy()

    df["USG"] = df["FGA"] + 0.44*df["FTA"] + df["TOV"]
    df["PTS_L5"] = df["PTS"].rolling(5).mean()
    df["MIN_L5"] = df["MIN"].rolling(5).mean()
    df["FGA_L5"] = df["FGA"].rolling(5).mean()

    df = df.dropna()

    FEATURES = ["MIN","FGA","FG3A","FTA","USG","PTS_L5","MIN_L5","FGA_L5"]

    return df, FEATURES


# -------------------------
# ML TRAIN
# -------------------------

def ml_projection(df, FEATURES, opp):

    split = int(len(df)*0.8)

    train = df.iloc[:split]
    valid = df.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=350,
        max_depth=7,
        random_state=1
    )

    model.fit(train[FEATURES], train["PTS"])

    pred = model.predict(valid[FEATURES])
    mae = mean_absolute_error(valid["PTS"], pred)

    row = df.iloc[-1].copy()

    row["USG"] = row["FGA"] + 0.44*row["FTA"] + row["TOV"]
    row["PTS_L5"] = df["PTS"].tail(5).mean()
    row["MIN_L5"] = df["MIN"].tail(5).mean()
    row["FGA_L5"] = df["FGA"].tail(5).mean()

    factor = matchup_factor(opp)
    row["FGA"] *= factor
    row["FG3A"] *= factor
    row["FTA"] *= factor

    X = row[FEATURES].values.reshape(1,-1)

    pred_next = model.predict(X)[0]

    pred_next *= fatigue_factor(df)

    return pred_next, mae


# -------------------------
# VARIANCE SCORE
# -------------------------

def variance_score(df):

    last = df["PTS"].tail(12)

    cv = last.std() / last.mean()

    if cv < 0.25:
        return "LOW"
    if cv < 0.40:
        return "MEDIUM"
    return "HIGH"


# -------------------------
# MONTE CARLO
# -------------------------

def simulate(mu, line, err):

    sims = 9000
    sd = max(err, mu*0.2)

    pts = np.random.normal(mu, sd, sims)
    pts = np.clip(pts,0,90)

    prob = (pts > line).mean()

    return prob, np.percentile(pts,10), np.percentile(pts,90)


# -------------------------
# UI
# -------------------------

st.title("NBA Hybrid Production Model")

name = st.text_input("Player name")
opp = st.text_input("Opponent")
line = st.number_input("Points line", value=20.5)

if st.button("Run Model"):

    df = load_player_log(name)

    if df is None or len(df) < MIN_GAMES_REQUIRED:
        st.write("Not enough data")
        st.stop()

    formula_pred = formula_projection(df, opp)

    if ML_AVAILABLE:

        df_ml, FEATURES = build_features(df)
        ml_pred, mae = ml_projection(df_ml, FEATURES, opp)

        final_pred = 0.6*ml_pred + 0.4*formula_pred
        err = mae

        st.write("Model: ML + Formula")

    else:

        final_pred = formula_pred
        err = df["PTS"].tail(10).std()

        st.write("Model: Formula fallback (no sklearn)")

    prob, p10, p90 = simulate(final_pred, line, err)

    st.write("Projection:", round(final_pred,1))
    st.write("Over %:", round(prob*100,1))
    st.write("Range:", round(p10,1), "-", round(p90,1))
    st.write("Variance:", variance_score(df))

    if prob > 0.75:
        st.success("STRONG OVER")
    elif prob > 0.60:
        st.info("LEAN OVER")
    elif prob < 0.40:
        st.warning("LEAN UNDER")
    else:
        st.write("NO EDGE")
