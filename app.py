import streamlit as st
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

from model_core import *

st.title("NBA Points ML Analyzer — Level 5")


# -------------------------
# PLAYER LOOKUP
# -------------------------

name = st.text_input("Player name")

def find_player_id(name):

    res = players.find_players_by_full_name(name)

    if not res:
        return None

    return res[0]["id"]


# -------------------------
# LOAD DATA
# -------------------------

@st.cache_data(ttl=3600)
def load_games(pid):

    gl = playergamelog.PlayerGameLog(
        player_id=pid,
        season="2024-25"
    )

    df = gl.get_data_frames()[0]

    if len(df) == 0:
        return None

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE")

    return df


# -------------------------
# RUN MODEL
# -------------------------

def run_full_model(pid, line):

    raw = load_games(pid)

    if raw is None or len(raw) < 20:
        return None

    df = build_features(raw)

    if len(df) < 20:
        return None

    model, feats = train_model(df)

    if model is None:
        return None

    pred = predict_next(df, model, feats)

    std = df["PTS"].tail(10).std()
    if pd.isna(std):
        std = 4

    over = prob_over(pred, line, std)

    hitrate = backtest_hit_rate(df, line)

    edge = pred - line

    vol = float(std)
    cons = float(df["PTS"].tail(10).std())

    conf = confidence_score(edge, over, hitrate, vol)

    return {
        "pred": pred,
        "over": over,
        "edge": edge,
        "conf": conf,
        "signal": signal_tag(over, edge),
        "stake": stake_size(conf),
        "vol": vol,
        "cons": cons,
        "curve": line_curve(pred, std)
    }


# -------------------------
# SINGLE PLAYER MODE
# -------------------------

line = st.number_input("Line", value=20.5)

if st.button("RUN SINGLE"):

    pid = find_player_id(name)

    if not pid:
        st.error("Player not found")
    else:

        r = run_full_model(pid, line)

        if not r:
            st.error("Not enough data")
        else:

            st.subheader("Prediction")

            st.write(f"Projection: {r['pred']:.2f}")
            st.write(f"Over probability: {r['over']*100:.1f}%")
            st.write(f"Edge: {r['edge']:.2f}")
            st.write(f"Confidence: {r['conf']:.1f}%")
            st.write(f"Signal: {r['signal']}")
            st.write(f"Stake %: {r['stake']}")

            st.subheader("Line sensitivity")

            st.dataframe(pd.DataFrame(r["curve"]))


# -------------------------
# TOP PICKS MODE
# -------------------------

if st.button("TOP PICKS SAMPLE"):

    sample = players.get_players()[:25]

    rows = []

    for p in sample:

        pid = p["id"]

        r = run_full_model(pid, line)

        if r:

            rows.append({
                "Player": p["full_name"],
                "Proj": round(r["pred"], 1),
                "OverProb": round(r["over"], 2),
                "Edge": round(r["edge"], 2),
                "Conf": round(r["conf"], 1),
                "Signal": r["signal"],
                "Stake": r["stake"]
            })

    if rows:

        df = pd.DataFrame(rows)
        df = df.sort_values("Conf", ascending=False)

        st.dataframe(df)
