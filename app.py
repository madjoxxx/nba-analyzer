import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import poisson
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats

# --- 1. NAPREDNE MATEMATIČKE FUNKCIJE ---
def calculate_weighted_ppm(log):
    """Računa poene po minuti dajući 70% težine zadnjim 5 utakmicama."""
    if len(log) < 5: return (log['PTS'] / log['MIN'].replace(0,1)).mean()
    recent = log.head(5)
    season = log.tail(len(log)-5)
    ppm_recent = (recent['PTS'] / recent['MIN'].replace(0,1)).mean()
    ppm_season = (season['PTS'] / season['MIN'].replace(0,1)).mean()
    return (ppm_recent * 0.7) + (ppm_season * 0.3)

def get_home_away_factor(log, is_home):
    """Prilagođava projekciju na temelju toga igra li igrač doma ili vani."""
    try:
        home_games = log[log['MATCHUP'].str.contains('vs.')]
        away_games = log[log['MATCHUP'].str.contains('@')]
        
        h_avg = home_games['PTS'].mean()
        a_avg = away_games['PTS'].mean()
        
        if is_home:
            return h_avg / log['PTS'].mean() if h_avg > 0 else 1.0
        else:
            return a_avg / log['PTS'].mean() if a_avg > 0 else 1.0
    except: return 1.0

# --- 2. UI I POSTAVKE ---
st.set_page_config(page_title="NBA BEAST V9.0", layout="wide")
st.title("🔥 NBA AI Beast V9.0 (Max Precision)")

nba_teams = sorted([team['nickname'] for team in teams.get_teams()])
positions = ["PG", "SG", "SF", "PF", "C"]

if 'batch_list' not in st.session_state:
    st.session_state.batch_list = []

with st.sidebar:
    st.header("🏀 Unos Parametara")
    ime = st.text_input("Ime igrača")
    poz = st.selectbox("Pozicija", positions)
    tim = st.selectbox("Njegov tim", nba_teams)
    protivnik = st.selectbox("Protivnik", nba_teams)
    lokacija = st.radio("Gdje se igra?", ["Doma", "Vani"])
    granica = st.number_input("Granica", value=15.5)
    spread = st.slider("Očekivana razlika (Spread)", 0, 20, 5)
    
    if st.button("➕ Dodaj na listu"):
        st.session_state.batch_list.append({
            "Ime": ime, "Poz": poz, "Tim": tim, "Protivnik": protivnik, 
            "Granica": granica, "Doma": lokacija == "Doma", "Spread": spread
        })
        st.rerun()

    if st.button("🗑️ Resetiraj"):
        st.session_state.batch_list = []
        st.rerun()

# --- 3. PRO ANALIZA ---
if st.session_state.batch_list:
    st.dataframe(pd.DataFrame(st.session_state.batch_list))
    
    if st.button("🚀 POKRENI BEAST MODE ANAL
