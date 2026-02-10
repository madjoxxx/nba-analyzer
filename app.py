import streamlit as st
import pandas as pd

from nba_api.stats.static import players
from model_core import *

st.set_page_config(layout="wide")

st.title("NBA PTS PROP — PRO SCANNER (15 Player Mode)")


# -------------------------
# CACHE PLAYER DB
# -------------------------

@st.cache_data
def load_players():
    return players.get_players()

ALL_PLAYERS = load_players()


# -------------------------
# SAFE FIND
# -------------------------

def find_player_id(name):

    
    import difflib

def find_player_id(name):

    if not name:
        return None

    name = name.lower().strip()

    names = [p["full_name"] for p in ALL_PLAYERS]

    matches = difflib.get_close_matches(
        name,
        names,
        n=3,
        cutoff=0.6
    )

    if matches:

        best = matches[0]

        for p in ALL_PLAYERS:
            if p["full_name"] == best:
                return p["id"], p["full_name"]

    # partial fallback
    for p in ALL_PLAYERS:
        if name in p["full_name"].lower():
            return p["id"], p["full_name"]

    return None


# -------------------------
# QUALITY SCORE
# -------------------------

def quality_score(row):

    score = 0

    score += row["Confidence"] * 0.5
    score += row["OverProb%"] * 0.3
    score += min(row["Edge"] * 10, 30) * 0.2

    if row["Vol"] == "LOW":
        score += 5

    if row["Cons%"] > 70:
        score += 5

    return round(score, 1)


def classify(row):

    if (
        row["Confidence"] >= 65 and
        row["Edge"] >= 2 and
        row["OverProb%"] >= 60 and
        row["Vol"] != "HIGH" and
        row["Cons%"] >= 60
    ):
        return "ELITE"

    if row["Confidence"] >= 60 and row["Edge"] >= 1.5:
        return "PLAYABLE"

    return "PASS"


# -------------------------
# INPUT GRID — 15 PLAYERS
# -------------------------

st.subheader("Enter up to 15 players")

cols = st.columns(3)

inputs = []

for i in range(15):

    with cols[i % 3]:

        pname = st.text_input(
            f"Player {i+1}",
            key=f"p{i}"
        )

        pline = st.number_input(
            f"Line {i+1}",
            value=20.5,
            key=f"l{i}"
        )

        inputs.append((pname, pline))


# -------------------------
# RUN SCAN
# -------------------------

if st.button("RUN AUTO SCAN (15)"):

    results = []
    prog = st.progress(0)

    total = len(inputs)

    for i, (name, line) in enumerate(inputs):

        prog.progress((i+1)/total)

        if not name or len(name) < 3:
            continue

        pid_data = find_player_id(name)

        if not pid_data:
            st.warning(f"Not found: {name}")
            continue

        pid, real_name = pid_data

        try:

            df = get_games(pid)

            if df is None or len(df) < 20:
                st.warning(f"No data: {real_name}")
                continue

            pred, feats = project(df)

            if pred is None:
                continue

            over = prob_over(pred, line, feats)
            hit = backtest(df, line)

            conf = confidence(over, hit)
            edge = pred - line

            row = {
                "Player": real_name,
                "Line": line,
                "Projection": round(pred,1),
                "Pick": "OVER" if over > 0.55 else "UNDER",
                "OverProb%": round(over*100,1),
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
            st.warning(f"Error for {real_name}")
            continue

    prog.empty()

    if not results:
        st.error("No valid players processed")
        st.stop()

    df_out = pd.DataFrame(results)

    elite = df_out[df_out["Tier"] == "ELITE"].sort_values("Score", ascending=False)
    playable = df_out[df_out["Tier"] == "PLAYABLE"].sort_values("Score", ascending=False)
    passed = df_out[df_out["Tier"] == "PASS"].sort_values("Score", ascending=False)


    # -------------------------
    # OUTPUT
    # -------------------------

    st.subheader("🟢 ELITE PICKS")
    st.dataframe(elite, use_container_width=True)

    st.subheader("🟡 PLAYABLE PICKS")
    st.dataframe(playable, use_container_width=True)

    with st.expander("🔴 FILTERED OUT"):
        st.dataframe(passed, use_container_width=True)
