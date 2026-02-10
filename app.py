import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog

# =========================
# SAFE IMPORT ML
# =========================

ML_AVAILABLE = True

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
except:
    ML_AVAILABLE = False


# =========================
# DATA FETCH
# =========================

@st.cache_data(ttl=3600)
def get_player_games(player_id, season="2024-25"):
    df = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season
    ).get_data_frames()[0]
    return df


# =========================
# FEATURE ENGINEERING
# =========================

def build_features(df):

    df = df.sort_values("GAME_DATE")

    df["PTS_roll5"] = df["PTS"].rolling(5).mean()
    df["PTS_roll10"] = df["PTS"].rolling(10).mean()
    df["MIN_roll5"] = df["MIN"].rolling(5).mean()
    df["FG3M_roll5"] = df["FG3M"].rolling(5).mean()
    df["FGA_roll5"] = df["FGA"].rolling(5).mean()

    df["HOME"] = df["MATCHUP"].str.contains("vs").astype(int)

    df = df.dropna()

    features = [
        "PTS_roll5",
        "PTS_roll10",
        "MIN_roll5",
        "FG3M_roll5",
        "FGA_roll5",
        "HOME"
    ]

    X = df[features]
    y = df["PTS"]

    return X, y


# =========================
# FALLBACK MODEL
# =========================

def fallback_prediction(df):

    last5 = df["PTS"].tail(5)

    mean = last5.mean()
    std = last5.std()

    return mean, std


# =========================
# ML TRAIN
# =========================

def train_ml(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = np.mean(np.abs(preds - y_test))

    return model, mae


# =========================
# MONTE CARLO
# =========================

def monte_carlo(mean, std, line, sims=10000):

    samples = np.random.normal(mean, std, sims)

    over = np.mean(samples > line)
    under = 1 - over

    return over, under


# =========================
# STREAMLIT UI
# =========================

st.title("NBA Points ML Predictor")

player_id = st.number_input("Player ID", value=203999)
line = st.number_input("Points Line", value=24.5)

if st.button("Run Prediction"):

    df = get_player_games(player_id)

    if len(df) < 15:
        st.error("Not enough games")
        st.stop()

    # ---------- fallback baseline ----------
    base_mean, base_std = fallback_prediction(df)

    # ---------- ML ----------
    if ML_AVAILABLE:

        X, y = build_features(df)

        model, mae = train_ml(X, y)

        latest = X.iloc[-1:]
        ml_pred = model.predict(latest)[0]

    else:
        ml_pred = base_mean
        mae = base_std

    # ---------- HYBRID ----------
    hybrid = ml_pred * 0.7 + base_mean * 0.3

    # ---------- monte carlo ----------
    over, under = monte_carlo(hybrid, base_std, line)

    # ---------- confidence ----------
    confidence = max(0, 100 - mae * 3)

    # =========================
    # OUTPUT
    # =========================

    st.subheader("Prediction")

    st.write(f"Baseline Avg (L5): {base_mean:.2f}")
    st.write(f"ML Prediction: {ml_pred:.2f}")
    st.write(f"Hybrid Prediction: {hybrid:.2f}")

    st.subheader("Probabilities")

    st.write(f"Over {line}: {over*100:.1f}%")
    st.write(f"Under {line}: {under*100:.1f}%")

    st.subheader("Model Confidence")

    st.write(f"{confidence:.1f}%")

    if confidence > 70:
        st.success("High confidence")
    elif confidence > 55:
        st.warning("Medium confidence")
    else:
        st.error("Low confidence")

    if not ML_AVAILABLE:
        st.warning("Running without sklearn — fallback mode active")
