import streamlit as st
import pandas as pd

from nba_api.stats.static import players

from model_core import *


st.title("NBA PTS ML Analyzer — Level 6")


# -------------------------
# PLAYER LOOKUP
# -------------------------

name = st.text_input("Player name")

player_id = None

if name:
    matches = players.find_players_by_full_name(name)

    if matches:
        player_id = matches[0]["id"]
        st.write("Player ID:", player_id)


line = st.number_input("Points line", value=20.5)


# -------------------------
# SINGLE MODE
# -------------------------

if st.button("RUN SINGLE"):

    if not player_id:
        st.warning("Enter player name")
    else:

        df = get_games(player_id)

        if len(df) < 20:
            st.warning("Not enough games")
        else:

            pred, feats = project_points(df)

            over = prob_over(pred, line, feats)

            hit = backtest_hit_rate(df, line)

            conf = confidence_score(over, hit)

            edge = pred - line

            vol = volatility_label(df)

            stake = stake_size(edge, conf)

            st.subheader("Prediction")

            st.write("Projection:", round(pred,2))
            st.write("Over probability:", round(over*100,1), "%")
            st.write("Edge:", round(edge,2))
            st.write("Confidence:", conf)
            st.write("Volatility:", vol)
            st.write("Stake %:", stake)

            curve = []

            for l in [line-2, line-1, line, line+1, line+2]:
                curve.append({
                    "Line": l,
                    "OverProb": prob_over(pred, l, feats)
                })

            st.subheader("Line Sensitivity")
            st.dataframe(pd.DataFrame(curve))


# -------------------------
# TOP PICKS
# -------------------------

if st.button("RUN TOP PICKS"):

    sample = players.get_players()[:25]

    rows = []

    for p in sample:

        try:

            df = get_games(p["id"])

            if len(df) < 20:
                continue

            pred, feats = project_points(df)

            line_guess = df["PTS"].mean()

            over = prob_over(pred, line_guess, feats)

            hit = backtest_hit_rate(df, line_guess)

            conf = confidence_score(over, hit)

            edge = pred - line_guess

            stake = stake_size(edge, conf)

            rows.append({
                "Player": p["full_name"],
                "Proj": round(pred,1),
                "LineGuess": round(line_guess,1),
                "OverProb": round(over*100,1),
                "Edge": round(edge,1),
                "Conf": conf,
                "Stake": stake
            })

        except:
            continue

    df_out = pd.DataFrame(rows)

    df_out = df_out.sort_values(
        "Conf",
        ascending=False
    )

    st.dataframe(df_out)
