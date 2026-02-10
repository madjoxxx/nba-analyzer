import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog, teamgamelog

ML_AVAILABLE = True

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
except:
    ML_AVAILABLE = False


# =========================
# DATA
# =========================

@st.cache_data(ttl=3600)
def get_player_games(player_id):
    return playergamelog.PlayerGameLog(
        player_id=player_id,
        season="2024-25"
    ).get_data_frames()[0]


@st.cache_data(ttl=3600)
def get_team_games(team_id):
    return teamgamelog.TeamGameLog(
        team_id=team_id,
        season="2024-25"
    ).get_data_frames()[0]


# =========================
# FEATURES
# =========================

def add_features(df):

    df = df.sort_values("GAME_DATE")

    df["PTS_roll5"] = df["PTS"].rolling(5).mean()
    df["PTS_roll10"] = df["PTS"].rolling(10).mean()
    df["MIN_roll5"] = df["MIN"].rolling(5).mean()

    df["USAGE_PROXY"] = df["FGA"] + df["FTA"] * 0.44

    df["HOME"] = df["MATCHUP"].str.contains("vs").astype(int)

    df = df.dropna()

    feats = [
        "PTS_roll5",
        "PTS_roll10",
        "MIN_roll5",
        "USAGE_PROXY",
        "HOME"
    ]

    return df[feats], df["PTS"]


# =========================
# FALLBACK
# =========================

def baseline(df):
    last5 = df["PTS"].tail(5)
    return last5.mean(), last5.std()


# =========================
# ML
# =========================

def train_ml(X, y):

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=7,
        random_state=42
    )

    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    mae = np.mean(np.abs(pred - yte))

    return model, mae


# =========================
# MATCHUP DEFENSE
# =========================

def defense_adjustment(team_id):

    try:
        tdf = get_team_games(team_id)

        opp_pts = tdf["PTS"].tail(10).mean()

        league_avg = 114

        adj = opp_pts / league_avg

        return adj

    except:
        return 1.0


# =========================
# PACE APPROX
# =========================

def pace_adjustment(df):

    poss_proxy = df["FGA"] + df["FTA"] * 0.44 + df["TOV"]

    pace = poss_proxy.tail(5).mean()

    league = 100

    return pace / league


# =========================
# MONTE CARLO
# =========================

def monte(mean, std, line):

    sims = np.random.normal(mean, std, 12000)

    over = np.mean(sims > line)
    return over, 1-over


# =========================
# UI
# =========================

st.title("NBA Production ML Predictor")

player_id = st.number_input("Player ID", value=203999)
team_id = st.number_input("Opponent Team ID", value=1610612738)
line = st.number_input("Points Line", value=24.5)

if st.button("Run"):

    df = get_player_games(player_id)

    if len(df) < 15:
        st.error("Not enough games")
        st.stop()

    base_mean, base_std = baseline(df)

    # ----- ML -----
    if ML_AVAILABLE:

        X, y = add_features(df)
        model, mae = train_ml(X, y)
        ml_pred = model.predict(X.iloc[-1:])[0]

    else:
        ml_pred = base_mean
        mae = base_std

    # ----- matchup -----
    def_adj = defense_adjustment(team_id)

    # ----- pace -----
    pace_adj = pace_adjustment(df)

    # ----- hybrid -----
    pred = ml_pred * 0.65 + base_mean * 0.35

    pred *= def_adj
    pred *= pace_adj

    # ----- monte carlo -----
    over, under = monte(pred, base_std, line)

    # ----- edge -----
    edge = pred - line

    # ----- confidence -----
    conf = max(0, 100 - mae*3)

    # ================= OUTPUT =================

    st.subheader("Prediction")
    st.write(f"Baseline: {base_mean:.2f}")
    st.write(f"ML: {ml_pred:.2f}")
    st.write(f"Adjusted: {pred:.2f}")

    st.subheader("Probabilities")
    st.write(f"Over: {over*100:.1f}%")
    st.write(f"Under: {under*100:.1f}%")

    st.subheader("Edge Score")
    st.write(f"{edge:.2f} pts")

    if edge > 2:
        st.success("Strong value")
    elif edge > 0.7:
        st.warning("Small value")
    else:
        st.error("No edge")

    st.subheader("Model Confidence")
    st.write(f"{conf:.1f}%")

    if not ML_AVAILABLE:
        st.warning("Fallback mode — sklearn missing")
