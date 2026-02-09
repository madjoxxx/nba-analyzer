import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from scipy.stats import poisson
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats

# --- 1. POMOĆNE FUNKCIJE ---
def calculate_weighted_ppm(log):
    if len(log) < 5: return (log['PTS'] / log['MIN'].replace(0,1)).mean()
    recent = log.head(5)
    season = log.tail(len(log)-5)
    ppm_recent = (recent['PTS'] / recent['MIN'].replace(0,1)).mean()
    ppm_season = (season['PTS'] / season['MIN'].replace(0,1)).mean()
    return (ppm_recent * 0.7) + (ppm_season * 0.3)

def check_schedule_fatigue(log):
    """Blaži kriterij: Samo ekstremni umor (3 u 4 dana)."""
    try:
        if len(log) < 3: return 1.0
        log['GAME_DATE_DT'] = pd.to_datetime(log['GAME_DATE'])
        last_game = log.iloc[0]['GAME_DATE_DT']
        three_games_ago = log.iloc[2]['GAME_DATE_DT']
        diff = (last_game - three_games_ago).days
        
        # Samo ako je baš 3 utakmice u 4 dana (ekstremni umor)
        if diff <= 4:
            return 0.94 # Blaži penal od 6%
        return 1.0
    except: return 1.0

def get_dvp_2_0(opponent_team, position):
    rim_protectors = ["Timberwolves", "Celtics", "Lakers", "Thunder", "Cavaliers"]
    perimeter_defenders = ["Heat", "Magic", "Knicks", "Pelicans", "76ers"]
    if position in ["C", "PF"] and any(t in opponent_team for t in rim_protectors):
        return 0.90
    if position in ["PG", "SG"] and any(t in opponent_team for t in perimeter_defenders):
        return 0.92
    return 1.0

@st.cache_data(ttl=3600)
def get_injury_report():
    try:
        url = "https://www.cbssports.com/nba/injuries/"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(response.text)
        df = pd.concat(tables, ignore_index=True)
        cols = ['Player', 'Position', 'Updated', 'Injury', 'Status']
        return df[[c for c in cols if c in df.columns]]
    except: return None

# --- 2. UI POSTAVKE ---
st.set_page_config(page_title="NBA AI BEAST V11.2", layout="wide")

with st.sidebar:
    st.header("🚑 Injury Watch")
    injuries = get_injury_report()
    if injuries is not None:
        search = st.text_input("Traži ozljedu")
        if search:
            st.dataframe(injuries[injuries['Player'].str.contains(search, case=False, na=False)], hide_index=True)
        else:
            st.dataframe(injuries.head(10), hide_index=True)

st.title("🔥 NBA AI Beast V11.2 (Optimized)")

if 'batch_list' not in st.session_state:
    st.session_state.batch_list = []
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# Unos igrača
with st.expander("➕ DODAJ NOVOG IGRAČA", expanded=not st.session_state.analysis_done):
    c1, c2, c3 = st.columns(3)
    with c1:
        ime = st.text_input("Ime igrača")
        poz = st.selectbox("Pozicija", ["PG", "SG", "SF", "PF", "C"])
    with c2:
        nba_teams = sorted([t['nickname'] for t in teams.get_teams()])
        tim = st.selectbox("Njegov tim", nba_teams)
        protivnik = st.selectbox("Protivnik", nba_teams)
    with c3:
        granica = st.number_input("Granica", value=18.5)
        lokacija = st.radio("Lokacija", ["Doma", "Vani"])
        spread = st.slider("Spread", 0, 20, 5)

    if st.button("➕ DODAJ NA LISTU"):
        if ime:
            st.session_state.batch_list.append({
                "Ime": ime, "Poz": poz, "Tim": tim, "Protivnik": protivnik, 
                "Granica": granica, "Doma": lokacija == "Doma", "Spread": spread
            })
            st.session_state.analysis_done = False
            st.rerun()

# --- 3. AKCIJSKE TIPKE ---
col_run, col_clear = st.columns(2)
with col_run:
    run_btn = st.button("🚀 POKRENI ANALIZU", use_container_width=True)
with col_clear:
    if st.button("🗑️ OBRIŠI SVE (CLEAR)", use_container_width=True):
        st.session_state.batch_list = []
        st.session_state.analysis_done = False
        st.rerun()

# --- 4. PRIKAZ REZULTATA ---
if st.session_state.batch_list:
    if run_btn:
        st.session_state.analysis_done = True
        
    if st.session_state.analysis_done:
        stats_df = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced').get_data_frames()[0]
        l_pace = stats_df['PACE'].mean()
        l_def = stats_df['DEF_RATING'].mean()

        for p in st.session_state.batch_list:
            try:
                p_search = players.find_players_by_full_name(p['Ime'])
                if not p_search: continue
                p_id = p_search[0]['id']
                log = playergamelog.PlayerGameLog(player_id=p_id, season='2024-25').get_data_frames()[0]

                fatigue_f = check_schedule_fatigue(log)
                dvp_f = get_dvp_2_0(p['Protivnik'], p['Poz'])
                w_ppm = calculate_weighted_ppm(log)
                base_min = log['MIN'].median()
                if p['Spread'] > 12: base_min *= 0.90
                
                t_stats = stats_df[stats_df['TEAM_NAME'].str.contains(p['Tim'])].iloc[0]
                o_stats = stats_df[stats_df['TEAM_NAME'].str.contains(p['Protivnik'])].iloc[0]
                pace_f = ((t_stats['PACE'] * o_stats['PACE']) / l_pace) / l_pace
                def_f = o_stats['DEF_RATING'] / l_def
                
                final_proj = base_min * w_ppm * pace_f * def_f * fatigue_f * dvp_f
                prob_over = (1 - poisson.cdf(p['Granica'] - 0.5, final_proj)) * 100
                confidence = int(max(0, min(100, 100 * (1 - (log['PTS'].std() / log['PTS'].mean())))))

                # Signalni sustav
                if prob_over > 65 and confidence > 60:
                    status = "🟢 ELITNI OVER"
                    expand = True
                elif prob_over < 35 and confidence > 60:
                    status = "🔵 ELITNI UNDER"
                    expand = True
                else:
                    status = "⚪ PROVJERI DETALJE"
                    expand = False

                with st.expander(f"{status} - {p['Ime']} ({round(prob_over, 1)}%)", expanded=expand):
                    colA, colB = st.columns([1, 2])
                    with colA:
                        st.metric("Prognoza", f"{round(final_proj, 1)} pts")
                        st.write(f"Conf: {confidence}%")
                        if fatigue_f < 1.0: st.warning("⚠️ Umoran (3 u 4)")
                    with colB:
                        last_10 = log.head(10).iloc[::-1]
                        st.line_chart(pd.DataFrame({'PTS': last_10['PTS'].values, 'Line': [p['Granica']]*10}))
            except: continue
