import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from scipy.stats import poisson
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats

# --- 1. NAPREDNE MATEMATIČKE FUNKCIJE ---
def calculate_weighted_ppm(log):
    if len(log) < 5: return (log['PTS'] / log['MIN'].replace(0,1)).mean()
    recent = log.head(5)
    season = log.tail(len(log)-5)
    ppm_recent = (recent['PTS'] / recent['MIN'].replace(0,1)).mean()
    ppm_season = (season['PTS'] / season['MIN'].replace(0,1)).mean()
    return (ppm_recent * 0.7) + (ppm_season * 0.3)

def get_home_away_factor(log, is_home):
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

# --- 2. INJURY WATCH (ANTI-BLOCK VERZIJA) ---
@st.cache_data(ttl=3600)
def get_injury_report():
    try:
        url = "https://www.cbssports.com/nba/injuries/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(response.text)
        # CBS obično ima jednu tablicu po timu, pa ih sve spajamo
        df = pd.concat(tables, ignore_index=True)
        # Pokušavamo zadržati samo bitne kolone ako postoje
        cols_to_keep = ['Player', 'Position', 'Updated', 'Injury', 'Status']
        available_cols = [c for c in cols_to_keep if c in df.columns]
        return df[available_cols]
    except Exception as e:
        st.sidebar.warning(f"Problem s izvorom ozljeda: {e}")
        return None

# --- 3. UI POSTAVKE ---
st.set_page_config(page_title="NBA AI BEAST V10.1", layout="wide")

# Sidebar: Injury Watch
with st.sidebar:
    st.header("🚑 Injury Watch")
    injuries = get_injury_report()
    if injuries is not None:
        search_player = st.text_input("Traži ozljedu (npr. Embiid)")
        if search_player:
            filtered = injuries[injuries['Player'].str.contains(search_player, case=False, na=False)]
            st.dataframe(filtered, hide_index=True)
        else:
            st.caption("Zadnje objavljeno (Top 10):")
            st.dataframe(injuries.head(10), hide_index=True)
    else:
        st.error("Ozljede trenutno nedostupne.")

st.title("🏀 NBA AI Beast V10.1 (Ultimate Edition)")

nba_teams = sorted([team['nickname'] for team in teams.get_teams()])
positions = ["PG", "SG", "SF", "PF", "C"]

if 'batch_list' not in st.session_state:
    st.session_state.batch_list = []

# Unos igrača
with st.expander("➕ DODAJ NOVOG IGRAČA ZA ANALIZU", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        ime = st.text_input("Ime igrača")
        poz = st.selectbox("Pozicija", positions)
    with col2:
        tim = st.selectbox("Njegov tim", nba_teams)
        protivnik = st.selectbox("Protivnik", nba_teams)
    with col3:
        granica = st.number_input("Granica", value=15.5)
        lokacija = st.radio("Lokacija", ["Doma", "Vani"])
        spread = st.slider("Spread (Razlika)", 0, 20, 5)
    
    if st.button("➕ DODAJ NA LISTU"):
        if ime:
            st.session_state.batch_list.append({
                "Ime": ime, "Poz": poz, "Tim": tim, "Protivnik": protivnik, 
                "Granica": granica, "Doma": lokacija == "Doma", "Spread": spread
            })
            st.rerun()

# --- 4. ANALIZA I GRAFIKONI ---
if st.session_state.batch_list:
    st.subheader("📋 Igrači spremni za Beast Mode")
    st.dataframe(pd.DataFrame(st.session_state.batch_list), use_container_width=True)
    
    if st.button("🚀 POKRENI ULTIMATE ANALIZU"):
        stats_df = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced').get_data_frames()[0]
        l_pace = stats_df['PACE'].mean()
        l_def = stats_df['DEF_RATING'].mean()

        for p in st.session_state.batch_list:
            try:
                p_search = players.find_players_by_full_name(p['Ime'])
                if not p_search:
                    st.error(f"Igrač {p['Ime']} nije pronađen.")
                    continue
                
                p_id = p_search[0]['id']
                log = playergamelog.PlayerGameLog(player_id=p_id, season='2024-25').get_data_frames()[0]
                if log.empty: continue

                # Kalkulacije (Beast Mode)
                base_min = log['MIN'].median()
                if p['Spread'] > 12: base_min *= 0.90 
                w_ppm = calculate_weighted_ppm(log)
                ha_factor = get_home_away_factor(log, p['Doma'])
                
                t_stats = stats_df[stats_df['TEAM_NAME'].str.contains(p['Tim'])].iloc[0]
                o_stats = stats_df[stats_df['TEAM_NAME'].str.contains(p['Protivnik'])].iloc[0]
                
                pace_f = ((t_stats['PACE'] * o_stats['PACE']) / l_pace) / l_pace
                def_f = o_stats['DEF_RATING'] / l_def
                vol_f = max(0.9, min(1.1, log.head(5)['FGA'].mean() / log.head(20)['FGA'].mean()))

                final_proj = base_min * w_ppm * pace_f * def_f * ha_factor * vol_f
                confidence = int(max(0, min(100, 100 * (1 - (log['PTS'].std() / log['PTS'].mean())))))
                prob_over = (1 - poisson.cdf(p['Granica'] - 0.5, final_proj)) * 100

                # --- PRIKAZ REZULTATA ---
                with st.container():
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.metric(label=f"🏀 {p['Ime']}", value=f"{round(final_proj, 1)} pts", delta=f"{round(final_proj - p['Granica'], 1)} vs Granica")
                        tip = "✅ OVER" if prob_over > 68 and confidence > 62 else ("❌ UNDER" if prob_over < 32 and confidence > 62 else "🚫 PASS")
                        st.subheader(f"TIP: {tip}")
                        st.write(f"Vjerojatnost: {round(prob_over, 1)}% | Confidence: {confidence}%")
                    
                    with c2:
                        last_10 = log.head(10).iloc[::-1]
                        chart_data = pd.DataFrame({
                            'Poeni': last_10['PTS'].values,
                            'Granica': [p['Granica']] * len(last_10)
                        })
                        st.line_chart(chart_data)
                    st.divider()

            except Exception as e:
                st.error(f"Greška kod {p['Ime']}: {e}")

    if st.button("🗑️ Očisti sve"):
        st.session_state.batch_list = []
        st.rerun()
