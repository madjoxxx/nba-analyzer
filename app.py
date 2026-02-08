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
    st.dataframe(pd.DataFrame(st.session_state.batch_list), use_container_width=True)
    
    if st.button("🚀 POKRENI BEAST MODE ANALIZU"):
        results = []
        with st.spinner('Beast Mode: Računam težinske prosjeke i home/away faktore...'):
            stats_df = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced').get_data_frames()[0]
            l_pace = stats_df['PACE'].mean()
            l_def = stats_df['DEF_RATING'].mean()

            for p in st.session_state.batch_list:
                try:
                    p_id = players.find_players_by_full_name(p['Ime'])[0]['id']
                    log = playergamelog.PlayerGameLog(player_id=p_id, season='2024-25').get_data_frames()[0]

                    # A) Napredna Minutaža (Blowout penal)
                    base_min = log['MIN'].median()
                    if p['Spread'] > 12: base_min *= 0.90 

                    # B) Težinska Efikasnost (Recent Form)
                    w_ppm = calculate_weighted_ppm(log)

                    # C) Home/Away Split
                    ha_factor = get_home_away_factor(log, p['Doma'])

                    # D) Matchup & Tempo
                    t_stats = stats_df[stats_df['TEAM_NAME'].str.contains(p['Tim'])].iloc[0]
                    o_stats = stats_df[stats_df['TEAM_NAME'].str.contains(p['Protivnik'])].iloc[0]
                    pace_f = ((t_stats['PACE'] * o_stats['PACE']) / l_pace) / l_pace
                    def_f = o_stats['DEF_RATING'] / l_def

                    # E) Volumen Šuta (Zadnjih 5 utakmica)
                    vol_f = max(0.9, min(1.1, log.head(5)['FGA'].mean() / log.head(20)['FGA'].mean()))

                    # --- KONAČNA FORMULA ---
                    final_proj = base_min * w_ppm * pace_f * def_f * ha_factor * vol_f
                    
                    # Confidence Score
                    confidence = int(max(0, min(100, 100 * (1 - (log['PTS'].std() / log['PTS'].mean())))))
                    prob_over = (1 - poisson.cdf(p['Granica'] - 0.5, final_proj)) * 100

                    results.append({
                        "Igrač": p['Ime'],
                        "Granica": p['Granica'],
                        "Proj": round(final_proj, 1),
                        "Over %": f"{round(prob_over, 1)}%",
                        "Conf": f"{confidence}%",
                        "TIP": "✅ OVER" if prob_over > 68 and confidence > 62 else ("❌ UNDER" if prob_over < 32 and confidence > 62 else "🚫 PASS")
                    })
                except Exception as e:
                    st.error(f"Greška kod {p.get('Ime', 'Nepoznato')}: {e}")

        if results:
            st.divider()
            st.subheader("📊 Beast Mode Rezultati")
            st.table(pd.DataFrame(results))
            vrijeme = datetime.now().strftime("%d.%m.%Y. u %H:%M:%S")
            st.caption(f"🕒 Analiza završena: {vrijeme}")
