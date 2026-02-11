import streamlit as st
import pandas as pd
import difflib

from nba_api.stats.static import players
from model_core import *

st.set_page_config(layout="wide")

st.title("NBA PTS PROP — ULTRA SCAN DESK")

@st.cache_data
def load_players():
    return players.get_players()

ALL_PLAYERS = load_players()
PLAYER_NAMES = sorted([p["full_name"] for p in ALL_PLAYERS])


def get_player_by_name(name):

    if not name:
        return None

    for p in ALL_PLAYERS:
        if p["full_name"] == name:
            return p["id"], p["full_name"]

    m = difflib.get_close_matches(name, PLAYER_NAMES, n=1, cutoff=0.6)

    if m:
        for p in ALL_PLAYERS:
            if p["full_name"] == m[0]:
                return p["id"], p["full_name"]

    return None


def quality_score(row):

    score = 0
    score += row["Confidence"] * 0.45
    score += row["Prob"] * 0.35
    score += min(abs(row["Edge"]) * 12, 30) * 0.2

    if row["Vol"] == "LOW":
        score += 5

    if row["Cons%"] > 70:
        score += 5

    return round(score,1)


def classify(row):

    # ELITE OVER
    if (
        row["Pick"] == "OVER" and
        row["Prob"] >= 62 and
        row["Confidence"] >= 60 and
        row["Edge"] >= 2
    ):
        return "ELITE"

    # ELITE UNDER — FIX
    if (
        row["Pick"] == "UNDER" and
        row["Prob"] >= 65 and
        row["Edge"] <= -2
    ):
        return "ELITE"

    if row["Confidence"] >= 55 and abs(row["Edge"]) >= 1.2:
        return "PLAYABLE"

    return "PASS"


st.subheader("15 Player Ultra Scanner")

cols = st.columns(3)
inputs = []

for i in range(15):

    with cols[i % 3]:

        pname = st.selectbox(
            f"Player {i+1}",
            [""] + PLAYER_NAMES,
            key=f"p{i}"
        )

        line = st.number_input(
            f"PTS Line {i+1}",
            value=20.5,
            key=f"l{i}"
        )

        inputs.append((pname, line))


if st.button("RUN ULTRA SCAN"):

    results = []

    for name, line in inputs:

        if not name:
            continue

        pid_data = get_player_by_name(name)

        if not pid_data:
            continue

        pid, real_name = pid_data

        df = get_games(pid)

        if df is None:
            continue

        pred, feats = project(df)

        if pred is None:
            continue

        over_prob = prob_over(pred, line, feats)
        under_prob = 1 - over_prob

        hit = backtest(df, line)

        conf = confidence(over_prob, hit)
        edge = pred - line

        pick = "OVER" if over_prob >= 0.5 else "UNDER"
        prob = over_prob if pick == "OVER" else under_prob

        row = {
            "Player": real_name,
            "Line": line,
            "Proj": round(pred,1),
            "Pick": pick,
            "Prob": round(prob*100,1),
            "Edge": round(edge,1),
            "Confidence": conf,
            "Vol": volatility(df),
            "Cons%": consistency(df),
            "Stake": stake(edge, conf)
        }

        row["Score"] = quality_score(row)
        row["Tier"] = classify(row)

        results.append(row)

    if not results:
        st.error("No valid players processed — try active rotation players")
        st.stop()

    df_out = pd.DataFrame(results)

    elite_over = df_out[(df_out.Pick=="OVER") & (df_out.Tier=="ELITE")]
    elite_under = df_out[(df_out.Pick=="UNDER") & (df_out.Tier=="ELITE")]
    playable = df_out[df_out.Tier=="PLAYABLE"]
    passed = df_out[df_out.Tier=="PASS"]

    st.subheader("🟢 ELITE OVER")
    st.dataframe(elite_over)

    st.subheader("🔵 ELITE UNDER")
    st.dataframe(elite_under)

    st.subheader("🟡 PLAYABLE")
    st.dataframe(playable)

    with st.expander("🔴 FILTERED"):
        st.dataframe(passed)
