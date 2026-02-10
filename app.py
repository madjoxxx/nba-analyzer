import streamlit as st
import pandas as pd
from nba_api.stats.static import players

from model_core import *

st.title("NBA PTS PROP SCANNER — Box Mode")


# -------------------------
# PLAYER FIND SAFE
# -------------------------

def safe_find_player(name):

    if not name or len(name) < 3:
        return None

    matches = players.find_players_by_full_name(name)

    if not matches:
        return None

    return matches[0]["id"], matches[0]["full_name"]


# -------------------------
# INPUT BOXES
# -------------------------

st.subheader("Enter Players + Lines")

cols = st.columns(2)

inputs = []

for i in range(5):

    with cols[i % 2]:

        pname = st.text_input(f"Player {i+1}", key=f"p{i}")
        pline = st.number_input(
            f"Line {i+1}",
            value=20.5,
            key=f"l{i}"
        )

        inputs.append((pname, pline))


# -------------------------
# RUN SCAN
# -------------------------

if st.button("RUN SCAN"):

    rows = []

    progress = st.progress(0)

    for idx, (name, line) in enumerate(inputs):

        progress.progress((idx+1)/len(inputs))

        pid_data = safe_find_player(name)

        if not pid_data:
            continue

        pid, real_name = pid_data

        try:

            df = get_games(pid)

            if len(df) < 25:
                continue

            pred, feats = project_points(df)

            if pred is None:
                continue

            over = prob_over(pred, line, feats)

            hit = backtest_hit_rate(df, line)

            conf = confidence(over, hit)

            edge = pred - line

            vol = volatility(df)

            cons = consistency(df)

            pick = "OVER" if over > 0.55 else "UNDER"

            rows.append({
                "Player": real_name,
                "Line": line,
                "Projection": round(pred,1),
                "Pick": pick,
                "OverProb%": round(over*100,1),
                "Edge": round(edge,1),
                "Confidence": conf,
                "Volatility": vol,
                "Consistency%": cons,
                "Stake": stake(edge, conf)
            })

        except:
            continue

    progress.empty()

    if len(rows) == 0:
        st.error("No valid players found — check spelling")
    else:

        out = pd.DataFrame(rows)

        out = out.sort_values(
            "Confidence",
            ascending=False
        )

        st.subheader("Scan Results")
        st.dataframe(out)
