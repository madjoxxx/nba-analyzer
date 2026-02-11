import streamlit as st
import pandas as pd
from nba_api.stats.static import players
from model_core import run_model

st.set_page_config(layout="wide")

st.title("🏀 NBA AI Prop Engine — Ultra UI")

# -----------------------------
# PLAYER LOOKUP
# -----------------------------

def get_pid(name):
    res = players.find_players_by_full_name(name)
    if res:
        return res[0]["id"]
    return None


# -----------------------------
# INPUT GRID
# -----------------------------

st.subheader("Player Scan Grid (max 15)")

rows = []

for i in range(15):
    c1,c2,c3 = st.columns([3,1,2])
    name = c1.text_input(f"Player {i+1}", key=f"n{i}")
    line = c2.text_input("Line", key=f"l{i}")
    opp = c3.text_input("Opponent (optional)", key=f"o{i}")

    if name and line:
        rows.append((name,line,opp))

# -----------------------------
# RUN
# -----------------------------

if st.button("RUN SCAN"):

    out = []

    for name,line,opp in rows:

        try:
            line = float(line)
        except:
            continue

        pid = get_pid(name)

        if not pid:
            continue

        r = run_model(pid,line,opp_override=opp if opp else None)

        if r:
            r["Player"] = name
            out.append(r)

    if not out:
        st.error("No valid players found")
        st.stop()

    df = pd.DataFrame(out)

    st.subheader("All Results")
    st.dataframe(df.sort_values("Edge", ascending=False))

    # -----------------------------
    # ELITE OVER
    # -----------------------------

    elite_over = df[df["PickType"]=="ELITE_OVER"]

    st.subheader("🔥 Elite Over Picks")
    st.dataframe(elite_over.sort_values("P_over_%",ascending=False))

    # -----------------------------
    # ELITE UNDER — FIXED
    # -----------------------------

    elite_under = df[df["PickType"]=="ELITE_UNDER"]

    st.subheader("🧊 Elite Under Picks")
    st.dataframe(elite_under.sort_values("P_under_%",ascending=False))
