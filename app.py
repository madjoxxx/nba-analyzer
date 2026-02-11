import streamlit as st
import pandas as pd
import difflib

from nba_api.stats.static import players
from model_core import *

st.set_page_config(layout="wide")

st.title("NBA PTS PROP — ULTRA SCAN DESK")


# -------------------------
# LOAD PLAYERS
# -------------------------

@st.cache_data
def load_players():
    return players.get_players()

ALL_PLAYERS = load_players()
PLAYER_NAMES = sorted([p["full_name"] for p in ALL_PLAYERS])


# -------------------------
# LOOKUP
# -------------------------

def get_player_by_name(name):

    if not name:
        return None

    for p in ALL_PLAYERS:
        if p["full_name"] == name:
            return p["id"], p["full_name"]

    # fuzzy fallback
    matches = difflib.get_close_matches(name, PLAYER_NAMES, n=1, cutoff=0.65)

    if matches:
        for p in ALL_PLAYERS:
            if p["full_name"] == matches[0]:
                return p["id"], p["full_name"]

    return None


# -------------------------
# SCORING
# -------------------------

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

    if (
        row["Confidence"] >= 65 and
        abs(row["Edge"]) >= 2 and
        row["Prob"] >= 60 and
        row["Vol"] != "HIGH" and
        row["Cons%"] >= 60
    ):
        return "ELITE"

    if row["Confidence"] >= 58 and abs(row["Edge"]) >= 1.3:
        return "PLAYABLE"

    return "PASS"


# -------------------------
# INPUT GRID
# -------------------------

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


# -------------------------
# RUN SCAN
# -------------------------

if st.button("RUN ULTRA SCAN"):

    results = []
    prog = st.progress(0)

    total = len(inputs)

    for i, (name, line) in enumerate(inputs):

        prog.progress((i+1)/total)

        if not name:
            continue

        pid_data = get_player_by_name(name)

        if not pid_data:
            st.warning(f"Not found: {name}")
            continue

        pid, real_name = pid_data

        try:

            df = get_games(pid)

            if df is None or len(df) < 20:
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

        except Exception as e:
            continue

    prog.empty()

    if not results:
        st.error("No valid players processed")
        st.stop()

    df_out = pd.DataFrame(results)


    # -------------------------
    # SPLIT OVER / UNDER
    # -------------------------

    elite_over = df_out[(df_out.Pick=="OVER") & (df_out.Tier=="ELITE")].sort_values("Score", ascending=False)
    elite_under = df_out[(df_out.Pick=="UNDER") & (df_out.Tier=="ELITE")].sort_values("Score", ascending=False)

    playable = df_out[df_out.Tier=="PLAYABLE"].sort_values("Score", ascending=False)
    passed = df_out[df_out.Tier=="PASS"].sort_values("Score", ascending=False)


    # -------------------------
    # OUTPUT UI
    # -------------------------

    st.subheader("🟢 ELITE OVER PICKS")
    st.dataframe(elite_over, use_container_width=True)

    st.subheader("🔵 ELITE UNDER PICKS")
    st.dataframe(elite_under, use_container_width=True)

    st.subheader("🟡 PLAYABLE PICKS")
    st.dataframe(playable, use_container_width=True)

    with st.expander("🔴 FILTERED OUT"):
        st.dataframe(passed, use_container_width=True)
