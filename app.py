import streamlit as st
import pandas as pd
from nba_api.stats.static import players

from model_core import *

st.title("NBA PTS PROP SCANNER — Advanced ML")


# -------------------------
# SCAN INPUT
# -------------------------

st.subheader("Scan list input")

st.write("Format: Player Name,Line")

text = st.text_area(
"""
Example:
Jimmy Butler,21.5
Jayson Tatum,27.5
Jalen Brunson,24.5
"""
)

# -------------------------
# SCAN MODE
# -------------------------

if st.button("RUN SCAN"):

    lines = text.strip().split("\n")

    rows = []

    for row in lines:

        try:

            name, line = row.split(",")
            line = float(line)

            pl = players.find_players_by_full_name(name)[0]
            pid = pl["id"]

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
                "Player": name,
                "Line": line,
                "Proj": round(pred,1),
                "Pick": pick,
                "OverProb%": round(over*100,1),
                "Edge": round(edge,1),
                "Conf": conf,
                "Vol": vol,
                "Cons%": cons,
                "Stake": stake(edge, conf)
            })

        except:
            continue

    out = pd.DataFrame(rows)

    if len(out) == 0:
        st.warning("No valid players")
    else:
        out = out.sort_values("Conf", ascending=False)

        st.subheader("Best Picks")
        st.dataframe(out)
