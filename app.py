import streamlit as st
import pandas as pd

from nba_api.stats.static import players
from model_core import *

st.set_page_config(layout="wide")

st.title("NBA PTS PROP — AUTO PICK SCANNER")


# -------------------------
# CACHE PLAYERS
# -------------------------

@st.cache_data
def load_players():
    return players.get_players()

ALL_PLAYERS = load_players()


def find_player_id(name):

    name = name.lower().strip()

    for p in ALL_PLAYERS:
        if name in p["full_name"].lower():
            return p["id"], p["full_name"]

    return None


# -------------------------
# INPUT GRID
# -------------------------

st.subheader("Players + Lines")

cols = st.columns(3)

inputs = []

for i in range(6):

    with cols[i % 3]:
        pname = st.text_input(f"Player {i+1}", key=f"p{i}")
        pline = st.number_input(f"Line {i+1}", value=20.5, key=f"l{i}")
        inputs.append((pname, pline))


# -------------------------
# QUALITY SCORE
# -------------------------

def quality_score(row):

    score = 0

    score += row["Confidence"] * 0.5
    score += row["OverProb%"] * 0.3
    score += min(row["Edge"]*10, 30) * 0.2

    if row["Vol"] == "LOW":
        score += 5

    if row["Cons%"] > 70:
        score += 5

    return round(score,1)


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
# RUN SCAN
# -------------------------

if st.button("RUN AUTO SCAN"):

    results = []

    prog = st.progress(0)

    for i, (name, line) in enumerate(inputs):

        prog.progress((i+1)/len(inputs))

        if not name or len(name) < 3:
            continue

        pid_data = find_player_id(name)

        if not pid_data:
            st.warning(f"Not found: {name}")
            continue

        pid, real_name = pid_data

        df = get_games(pid)

        if df is None or len(df) < 20:
            continue

        pred, feats = project(df)

        if pred is None:
            continue

        over = prob_over(pred, line, feats)
        hit = backtest(df, line)

        conf = confidence(over, hit)
        edge = pred - line

        pick = "OVER" if over > 0.55 else "UNDER"

        row = {
            "Player": real_name,
            "Line": line,
            "Projection": round(pred,1),
            "Pick": pick,
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

    prog.empty()

    if not results:
        st.error("No valid players processed")
        st.stop()

    df_out = pd.DataFrame(results)

    elite = df_out[df_out["Tier"] == "ELITE"].sort_values("Score", ascending=False)
    playable = df_out[df_out["Tier"] == "PLAYABLE"].sort_values("Score", ascending=False)
    passed = df_out[df_out["Tier"] == "PASS"].sort_values("Score", ascending=False)


    # -------------------------
    # OUTPUT SECTIONS
    # -------------------------

    st.subheader("🟢 ELITE PICKS")
    if len(elite):
        st.dataframe(elite, use_container_width=True)
    else:
        st.write("None")


    st.subheader("🟡 PLAYABLE PICKS")
    if len(playable):
        st.dataframe(playable, use_container_width=True)
    else:
        st.write("None")


    with st.expander("🔴 FILTERED OUT"):
        st.dataframe(passed, use_container_width=True)
