import streamlit as st
import pandas as pd
from nba_api.stats.static import players
from model_core import run_model

st.set_page_config(layout="wide")

st.title("🏀 NBA AI Prop Engine — Ultra UI+")

# -----------------------------
# LOAD PLAYER LIST
# -----------------------------

@st.cache_data
def load_players():
    plist = players.get_players()
    plist = sorted(plist, key=lambda x: x["full_name"])
    names = [p["full_name"] for p in plist]
    id_map = {p["full_name"]: p["id"] for p in plist}
    return names, id_map

PLAYER_NAMES, PLAYER_ID_MAP = load_players()

# -----------------------------
# GRID INPUT — 15 ROWS
# -----------------------------

st.subheader("🔍 Player Scan Grid (Select — no typing errors)")

rows = []

for i in range(15):

    c1, c2, c3 = st.columns([3,1,2])

    name = c1.selectbox(
        f"Player {i+1}",
        [""] + PLAYER_NAMES,
        key=f"p{i}"
    )

    line = c2.text_input("Line", key=f"l{i}")
    opp = c3.text_input("Opponent (optional)", key=f"o{i}")

    if name and line:
        rows.append((name,line,opp))

# -----------------------------
# RUN BUTTON
# -----------------------------

if st.button("🚀 RUN SCAN"):

    out = []
    failed = []

    progress = st.progress(0)
    total = len(rows)

    for idx, (name,line,opp) in enumerate(rows):

        progress.progress((idx+1)/max(total,1))

        try:
            line = float(line)
        except:
            failed.append(name)
            continue

        pid = PLAYER_ID_MAP.get(name)

        if not pid:
            failed.append(name)
            continue

        try:
            r = run_model(pid, line, opp_override=opp if opp else None)
        except Exception as e:
            failed.append(name)
            continue

        if r:
            r["Player"] = name
            out.append(r)
        else:
            failed.append(name)

    progress.empty()

    # -----------------------------
    # FAIL INFO
    # -----------------------------

    if failed:
        st.warning("⚠️ Not processed: " + ", ".join(failed))

    if not out:
        st.error("No valid players processed")
        st.stop()

    df = pd.DataFrame(out)

    # -----------------------------
    # SORT CORE TABLE
    # -----------------------------

    df = df.sort_values("Edge", ascending=False)

    st.subheader("📊 All Results")
    st.dataframe(df, use_container_width=True)

    # -----------------------------
    # ELITE OVER PICKS
    # -----------------------------

    elite_over = df[df["PickType"]=="ELITE_OVER"]

    st.subheader("🔥 Elite Over Picks")

    if len(elite_over):
        st.dataframe(
            elite_over.sort_values("P_over_%", ascending=False),
            use_container_width=True
        )
    else:
        st.info("No elite over picks")

    # -----------------------------
    # ELITE UNDER PICKS
    # -----------------------------

    elite_under = df[df["PickType"]=="ELITE_UNDER"]

    st.subheader("🧊 Elite Under Picks")

    if len(elite_under):
        st.dataframe(
            elite_under.sort_values("P_under_%", ascending=False),
            use_container_width=True
        )
    else:
        st.info("No elite under picks")

    # -----------------------------
    # BEST PICKS AUTO PANEL
    # -----------------------------

    st.subheader("⭐ Best Auto Picks")

    best = df[
        (df["PickType"].isin(["ELITE_OVER","ELITE_UNDER"])) &
        (df["Edge"] >= 6)
    ]

    if len(best):
        st.dataframe(best, use_container_width=True)
    else:
        st.info("No auto-qualified picks today")
