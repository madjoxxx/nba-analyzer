import streamlit as st
import pandas as pd

from nba_api.stats.static import players

from model_core import *

st.set_page_config(layout="wide")

st.title("NBA PTS PROP — PRO SCANNER")


# -------------------------
# CACHE PLAYER DB
# -------------------------

@st.cache_data
def load_players():
    return players.get_players()

ALL_PLAYERS = load_players()


# -------------------------
# SAFE SEARCH
# -------------------------

def find_player_id(name):

    name = name.lower().strip()

    for p in ALL_PLAYERS:
        if name in p["full_name"].lower():
            return p["id"], p["full_name"]

    return None


# -------------------------
# INPUT UI
# -------------------------

st.subheader("Enter Players")

cols = st.columns(3)

inputs = []

for i in range(6):

    with cols[i % 3]:
        pname = st.text_input(f"Player {i+1}", key=f"p{i}")
        pline = st.number_input(f"Line {i+1}", value=20.5, key=f"l{i}")
        inputs.append((pname, pline))


# -------------------------
# RUN
# -------------------------

if st.button("RUN SCAN"):

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
            st.warning(f"No data: {real_name}")
            continue

        pred, feats = project(df)

        if pred is None:
            continue

        over = prob_over(pred, line, feats)
        hit = backtest(df, line)

        conf = confidence(over, hit)
        edge = pred - line

        pick = "OVER" if over > 0.55 else "UNDER"

        results.append({
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
        })

    prog.empty()

    if len(results) == 0:
        st.error("No valid players processed")
    else:

        out = pd.DataFrame(results)

        out = out.sort_values("Confidence", ascending=False)

        st.subheader("Best Props")
        st.dataframe(out, use_container_width=True)
