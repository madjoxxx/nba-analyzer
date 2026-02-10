import streamlit as st
import pandas as pd
import numpy as np

from nba_api.stats.endpoints import (
    playergamelog,
    commonallplayers
)

# =========================
# SAFE ML IMPORT
# =========================

ML_AVAILABLE = True
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
except:
    ML_AVAILABLE = False


# =========================
# CACHE DATA
# =========================

@st.cache_data(ttl=86400)
def load_players():
    return commonallplayers.CommonAllPlayers(
        is_only_current_season=1
    ).get_data_frames()[0]


@st.cache_data(ttl=3600)
def get_games(player_id):
    return playergamelog.PlayerGameLog(
        player_id=player_id,
        season="2024-25"
    ).get_data_frames()[0]


# =========================
# FEATURES
# =========================

def build_features(df):

    df = df.sort_values("GAME_DATE")

    df["PTS_r5"] = df["PTS"].rolling(5).mean()
    df["PTS_r10"] = df["PTS"].rolling(10).mean()
    df["MIN_r5"] = df["MIN"].rolling(5).mean()
    df["USAGE"] = df["FGA"] + df["FTA"] * 0.44
    df["HOME"] = df["MATCHUP"].str.contains("vs").astype(int)

    df = df.dropna()

    feats = ["PTS_r5", "PTS_r10", "MIN_r5", "USAGE", "HOME"]

    return df[feats], df["PTS"]


# =========================
# BASELINE
# =========================

def baseline(df):
    last5 = df["PTS"].tail(5)
    return last5.mean(), last5.std()


# =========================
# ML TRAIN
# =========================

def train_ml(X, y):

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=350,
        max_depth=7,
        random_state=42
    )

    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    mae = np.mean(np.abs(pred - yte))

    return model, mae


# =========================
# MONTE CARLO
# =========================

def monte(mean, std, line):

    sims = np.random.normal(mean, std, 12000)
    over = np.mean(sims > line)

    return over


# =========================
# PLAYER MODEL PIPELINE
# =========================

def run_model(player_id, line):

    try:
        df = get_games(player_id)
    except:
        return None

    if len(df) < 15:
        return None

    base_mean, base_std = baseline(df)

    if ML_AVAILABLE:

        X, y = build_features(df)
        model, mae = train_ml(X, y)
        ml_pred = model.predict(X.iloc[-1:])[0]

    else:
        ml_pred = base_mean
        mae = base_std

    pred = ml_pred * 0.65 + base_mean * 0.35

    over_prob = monte(pred, base_std, line)

    edge = pred - line
    conf = max(0, 100 - mae * 3)

    return {
        "pred": pred,
        "over": over_prob,
        "edge": edge,
        "conf": conf
    }


# =========================
# UI
# =========================

st.title("NBA ML Prop Finder")

players_df = load_players()

mode = st.radio(
    "Mode",
    ["Single Player", "Top Picks Scanner"]
)

line = st.number_input("Points Line", value=20.5)


# =========================
# SINGLE PLAYER MODE
# =========================

if mode == "Single Player":

    name = st.text_input("Player name")

    filtered = players_df[
        players_df["DISPLAY_FIRST_LAST"]
        .str.contains(name, case=False, na=False)
    ]

    if len(filtered) > 0:

        choice = st.selectbox(
            "Select player",
            filtered["DISPLAY_FIRST_LAST"].values
        )

        pid = int(
            filtered[
                filtered["DISPLAY_FIRST_LAST"] == choice
            ]["PERSON_ID"].values[0]
        )

        if st.button("Run Prediction"):

            r = run_model(pid, line)

            if r is None:
                st.error("Not enough data")
            else:
                st.subheader("Prediction")

                st.write(f"Projected: {r['pred']:.2f}")
                st.write(f"Over {line}: {r['over']*100:.1f}%")
                st.write(f"Edge: {r['edge']:.2f}")
                st.write(f"Confidence: {r['conf']:.1f}%")

                if r["edge"] > 2:
                    st.success("Strong Value")
                elif r["edge"] > 0.7:
                    st.warning("Small Value")
                else:
                    st.error("No Edge")


# =========================
# TOP PICKS SCANNER
# =========================

else:

    sample_size = st.slider(
        "Number of players to scan",
        10, 120, 40
    )

    if st.button("Scan Players"):

        results = []

        sample = players_df.head(sample_size)

        progress = st.progress(0)

        for i, row in enumerate(sample.itertuples()):

            pid = row.PERSON_ID
            name = row.DISPLAY_FIRST_LAST

            r = run_model(pid, line)

            if r and r["conf"] > 55:

                results.append({
                    "Player": name,
                    "Proj": round(r["pred"], 1),
                    "Over%": round(r["over"]*100, 1),
                    "Edge": round(r["edge"], 2),
                    "Conf": round(r["conf"], 1)
                })

            progress.progress((i+1)/sample_size)

        if len(results) == 0:
            st.error("No valid picks")
        else:

            df = pd.DataFrame(results)

            df = df.sort_values(
                ["Edge", "Over%"],
                ascending=False
            )

            st.subheader("Top Picks")

            st.dataframe(df.head(15))


# =========================
# FOOTER
# =========================

if not ML_AVAILABLE:
    st.warning("Running fallback model — sklearn not installed")
