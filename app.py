import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats
from nba_api.stats.static import players
import traceback

st.set_page_config(page_title="NBA Points Predictor Pro", layout="wide")

# -----------------------------
# CACHE — TEAM STATS
# -----------------------------

@st.cache_data(ttl=3600)
def get_team_stats():
    df = leaguedashteamstats.LeagueDashTeamStats().get_data_frames()[0]
    return df[['TEAM_ID','TEAM_NAME','PACE','DEF_RATING','OFF_RATING']]

TEAM_STATS = get_team_stats()
LEAGUE_PACE = TEAM_STATS['PACE'].mean()
LEAGUE_DEF = TEAM_STATS['DEF_RATING'].mean()

# -----------------------------
# MINUTES PARSER
# -----------------------------

def parse_minutes_col(min_series: pd.Series) -> pd.Series:
    def parse_min(x):
        if pd.isna(x):
            return np.nan
        try:
            return float(x)
        except:
            pass
        if isinstance(x, str) and ":" in x:
            try:
                mm, ss = x.split(":")
                return float(mm) + float(ss)/60
            except:
                return np.nan
        return np.nan
    return min_series.apply(parse_min)

def safe_sort_log(log):
    log = log.copy()
    log['GAME_DATE'] = pd.to_datetime(log['GAME_DATE'])
    return log.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

def recency_weights(n):
    return np.exp(-np.arange(n)/4)

def compute_ppm_features(log):
    log = safe_sort_log(log)
    log['MIN'] = parse_minutes_col(log['MIN'])
    log = log.dropna(subset=['MIN'])
    log = log[log['MIN'] > 3]

    if len(log) == 0:
        return 0.8, 0.25

    ppm = log['PTS'] / log['MIN']
    w = recency_weights(len(ppm))

    wppm = np.average(ppm, weights=w)
    return float(wppm), float(ppm.std())

def predict_minutes(log):
    log = safe_sort_log(log)
    log['MIN'] = parse_minutes_col(log['MIN'])
    mins = log['MIN'].dropna()

    if len(mins) == 0:
        return 28

    last5 = mins.head(5)

    med = last5.median()
    if len(last5) >= 2:
        trend = last5.iloc[0] - last5.iloc[-1]
    else:
        trend = 0

    pred = med + trend * 0.25
    return float(np.clip(pred, 12, 40))

def fatigue_factor(log):
    log = safe_sort_log(log)
    if len(log) < 3:
        return 1.0

    d0 = log.loc[0,'GAME_DATE']
    d2 = log.loc[2,'GAME_DATE']

    if (d0 - d2).days <= 3:
        return 0.94

    return 1.0

def pace_factor(team, opp):
    try:
        t = TEAM_STATS[TEAM_STATS.TEAM_NAME == team].iloc[0]
        o = TEAM_STATS[TEAM_STATS.TEAM_NAME == opp].iloc[0]
        return np.sqrt(t.PACE * o.PACE) / LEAGUE_PACE
    except:
        return 1.0

def defense_factor(opp):
    try:
        d = TEAM_STATS[TEAM_STATS.TEAM_NAME == opp].iloc[0].DEF_RATING
        return d / LEAGUE_DEF
    except:
        return 1.0

# -----------------------------
# MONTE CARLO MODEL
# -----------------------------

def simulate_points(mu_minutes, mu_ppm, ppm_sigma, line):
    sims = 20000

    min_sigma = max(2.5, mu_minutes * 0.12)

    sim_minutes = np.random.normal(mu_minutes, min_sigma, sims)
    sim_minutes = np.clip(sim_minutes, 5, 48)

    sim_ppm = np.random.normal(mu_ppm, max(0.05, ppm_sigma), sims)
    sim_ppm = np.clip(sim_ppm, 0.2, 2.5)

    pts = sim_minutes * sim_ppm

    prob_over = (pts > line).mean() * 100
    mean_proj = pts.mean()
    p10 = np.percentile(pts, 10)
    p90 = np.percentile(pts, 90)

    return prob_over, mean_proj, p10, p90

# -----------------------------
# DATA FETCH
# -----------------------------

@st.cache_data(ttl=1800)
def get_player_log(player_name):
    pl = players.find_players_by_full_name(player_name)
    if not pl:
        return None
    pid = pl[0]['id']
    df = playergamelog.PlayerGameLog(player_id=pid).get_data_frames()[0]
    return df

# -----------------------------
# UI
# -----------------------------

st.title("NBA Points Predictor — Advanced Model")

st.write("Unesi igrače (ime, tim, protivnik, linija).")

rows = st.number_input("Broj igrača", 1, 20, 3)

inputs = []
for i in range(rows):
    c1,c2,c3,c4 = st.columns(4)
    name = c1.text_input(f"Igrač {i+1}")
    team = c2.text_input("Tim", key=f"team{i}")
    opp = c3.text_input("Protivnik", key=f"opp{i}")
    line = c4.number_input("Granica", value=20.5, key=f"line{i}")
    inputs.append((name,team,opp,line))

# -----------------------------
# RUN
# -----------------------------

if st.button("Analiziraj"):

    results = []

    for name,team,opp,line in inputs:

        if not name:
            continue

        try:
            log = get_player_log(name)

            if log is None or len(log) == 0:
                st.warning(f"Nema podataka za {name}")
                continue

            log = safe_sort_log(log)
            log['MIN'] = parse_minutes_col(log['MIN'])

            mu_minutes = predict_minutes(log)
            mu_ppm, ppm_sigma = compute_ppm_features(log)

            f_fat = fatigue_factor(log)
            f_pace = pace_factor(team, opp)
            f_def = defense_factor(opp)

            adj_ppm = mu_ppm * f_pace * f_def * f_fat

            prob, proj, p10, p90 = simulate_points(
                mu_minutes,
                adj_ppm,
                ppm_sigma,
                line
            )

            conf = np.clip(100 - ppm_sigma*60, 40, 95)

            results.append({
                "Igrač": name,
                "Linija": line,
                "Projekcija": round(proj,1),
                "P_over_pct": round(prob,1),
                "P10": round(p10,1),
                "P90": round(p90,1),
                "Minutes_pred": round(mu_minutes,1),
                "PPM_pred": round(adj_ppm,3),
                "Confidence": round(conf,1)
            })

        except Exception as e:
            st.error(f"Greška za {name}: {e}")
            st.text(traceback.format_exc())

    if results:

        df = pd.DataFrame(results).sort_values("P_over_pct", ascending=False)
        st.dataframe(df, use_container_width=True)

        st.subheader("Signal")

        for _, r in df.iterrows():

            p_over = r["P_over_pct"]

            if p_over >= 75:
                tag = "🔥 STRONG OVER"
            elif p_over >= 60:
                tag = "✅ OVER lean"
            elif p_over <= 40:
                tag = "❌ UNDER lean"
            else:
                tag = "⚖️ No edge"

            st.write(
                f"{r['Igrač']}: {tag} ({p_over}%) — proj {r['Projekcija']} "
                f"[{r['P10']}–{r['P90']}]"
            )
