from prop_engine import (
    ensemble_prediction,
    volatility_flag,
    line_sensitivity,
    bet_signal,
    stake_size
)

import streamlit as st
import pandas as pd

from nba_api.stats.endpoints import playergamelog, commonallplayers

from model_core import (
    build_features,
    baseline,
    train_ml,
    monte,
    predict_minutes,
    consistency_score,
    backtest_hit_rate
)

# =====================
# DATA
# =====================

@st.cache_data(ttl=86400)
def load_players():
    return commonallplayers.CommonAllPlayers(
        is_only_current_season=1
    ).get_data_frames()[0]


@st.cache_data(ttl=3600)
def get_games(pid):
    return playergamelog.PlayerGameLog(
        player_id=pid,
        season="2024-25"
    ).get_data_frames()[0]


# =====================
# PIPELINE
# =====================

def run_full_model(pid, line):

    df = get_games(pid).reset_index(drop=True)

    if len(df) < 15:
        return None

    base_mean, base_std = baseline(df)

    X, y = build_features(df)
    model, mae = train_ml(X, y)

    if model:
        ml_pred = model.predict(X.iloc[-1:])[0]
    else:
        ml_pred = base_mean

    pred = ensemble_prediction(ml_pred, base_mean)

    minutes = predict_minutes(df)
    minutes_factor = minutes / df["MIN"].tail(5).mean()

    pred *= minutes_factor

    over = monte(pred, base_std, line)

    edge = pred - line

    consistency = consistency_score(df)
    hitrate = backtest_hit_rate(df, line)

    confidence = max(0, 100 - mae*3)

    vol = volatility_flag(base_std)

    signal = bet_signal(edge, over, confidence, consistency)

    stake = stake_size(confidence, edge)

    curve = line_sensitivity(pred, base_std, line)

    return {
        "pred": pred,
        "over": over,
        "edge": edge,
        "conf": confidence,
        "cons": consistency,
        "hit": hitrate,
        "vol": vol,
        "signal": signal,
        "stake": stake,
        "curve": curve
    }


# =====================
# UI
# =====================

st.title("NBA LEVEL 4 PROP MODEL")

players = load_players()

mode = st.radio(
    "Mode",
    ["Single", "Top Picks"]
)

line = st.number_input("Points Line", value=20.5)


# =====================
# SINGLE
# =====================

if mode == "Single":

    name = st.text_input("Player name")

    filt = players[
        players["DISPLAY_FIRST_LAST"]
        .str.contains(name, case=False, na=False)
    ]

    if len(filt) > 0:

        choice = st.selectbox(
            "Select",
            filt["DISPLAY_FIRST_LAST"]
        )

        pid = int(
            filt[
                filt["DISPLAY_FIRST_LAST"] == choice
            ]["PERSON_ID"].values[0]
        )

        if st.button("Run"):

            r = run_full_model(pid, line)

            st.write(r)


# =====================
# TOP PICKS
# =====================

else:

    n = st.slider("Players to scan", 20, 150, 60)

    if st.button("Scan"):

        rows = []

        for row in players.head(n).itertuples():

            r = run_full_model(row.PERSON_ID, line)

            if r and r["conf"] > 55:

                rows.append({
                    "Player": row.DISPLAY_FIRST_LAST,
                    "Proj": round(r["pred"],1),
                    "Over%": round(r["over"]*100,1),
                    "Edge": round(r["edge"],2),
                    "Conf": round(r["conf"],1),
                    "Cons": r["cons"],
                    "HitRate": round(r["hit"]*100,1),
                    "Grade": r["grade"]
                })

        df = pd.DataFrame(rows)

        df = df.sort_values(
            ["Grade","Edge","Over%"],
            ascending=False
        )

        st.dataframe(df.head(15))
