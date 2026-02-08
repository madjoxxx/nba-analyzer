import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import poisson
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats

# --- 1. POMOĆNE FUNKCIJE ---
def check_b2b(log):
    try:
        if log.empty: return False
        last_game_str = log.iloc[0]['GAME_DATE']
        last_game_date = datetime.strptime(last_game_str, '%b %d, %Y').date()
        yesterday = datetime.now().date() - timedelta(days=1)
        return last_game_date == yesterday
    except: return False

def get_position_factor(opponent, position):
    elite_def = ["Timberwolves", "Celtics", "Heat", "76ers", "Thunder", "Magic"]
    bad_def = ["Wizards", "Pistons", "Hornets", "Spurs", "Trail Blazers"]
    
    if any(team in opponent for team in elite_def): return 0.90
    if any(team in opponent for team in bad_def): return 1.10
    return 1.0

# --- 2. UI POSTAVKE ---
st.set_page_config(page_title="NBA Pro Analytics v8.0", layout="wide")
st.title("🏀 NBA Pro Analytics (Precision & Confidence Mode)")

nba_teams = sorted([team['nickname'] for team in teams.get_teams()])
positions = ["PG", "SG", "SF", "PF", "C"]

if 'batch_list' not in st.session_state:
    st.session_state.batch_list = []

with st.sidebar:
    st.header("⚙️ Postavke")
    ime = st.text_input("Ime igrača", "Amen Thompson")
    pozicija = st.selectbox("Pozicija", positions)
    moj_tim = st.selectbox("Njegov tim", nba_teams)
    protivnik = st.selectbox("Protivnik", nba_teams)
    granica = st.number_input("Granica", value=13.5)
    
    if st.button("➕ Dodaj na listu"):
        st.session_state.batch_list.append({
            "ime": ime, "poz": pozicija, "tim": moj_tim, "protivnik": protivnik, "granica": granica
        })

    if st.button("🗑️ Očisti sve"):
        st.session_state.batch_list = []
        st.rerun()

# --- 3. GLAVNA ANALIZA ---
if st.session_state.batch_list:
    if st.button("🚀 POKRENI PRECIZNU ANALIZU"):
        results = []
        with st.spinner('Kalibriram model i računam volume trend...'):
            stats_df = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced').get_data_frames()[0]
            l_pace = stats_df['PACE'].mean()
            l_def = stats_df['DEF_RATING'].mean()

            for p in st.session_state.batch_list:
                try:
                    p_search = players.find_players_by_full_name(p['ime'])
                    if not p_search: continue
                    
                    log = playergamelog.PlayerGameLog(player_id=p_search[0]['id'], season='2024-25').get_data_frames()[0]
                    if log.empty: continue

                    # --- KALIBRACIJA PRECIZNOSTI ---
                    # 1. Median Minutes (otpornije na blowout)
                    proj_min = log['MIN'].median()
                    
                    # 2. Recent Volume Trend (Zadnjih 5 vs zadnjih 20 utakmica po broju šuteva)
                    recent_fga = log.head(5)['FGA'].mean()
                    season_fga = log.head(20)['FGA'].mean()
                    volume_factor = max(0.85, min(1.15, recent_fga / season_fga)) if season_fga > 0 else 1.0

                    # 3. Konstantnost (Confidence Score)
                    # Što je veća varijacija poena, to je niži confidence
                    pts_std = log.head(15)['PTS'].std()
                    pts_mean = log.head(15)['PTS'].mean()
                    cv = pts_std / pts_mean if pts_mean > 0 else 1
                    confidence = max(0, min(100, int(100 * (1 - cv))))

                    # 4. Standardni faktori (B2B, DvP, Pace, Def)
                    b2b_multiplier = 0.90 if check_b2b(log) else 1.0
                    dvp_multiplier = get_position_factor(p['protivnik'], p['poz'])
                    eff = (log.head(15)['PTS'] / log.head(15)['MIN'].replace(0, 1)).mean()
                    
                    t_stats = stats_df[stats_df['TEAM_NAME'].str.contains(p['tim'])].iloc[0]
                    o_stats = stats_df[stats_df['TEAM_NAME'].str.contains(p['protivnik'])].iloc[0]
                    pace_f = ((t_stats['PACE'] * o_stats['PACE']) / l_pace) / l_pace
                    def_f = o_stats['DEF_RATING'] / l_def

                    # --- KONAČNA PROJEKCIJA ---
                    adj_mu = proj_min * eff * pace_f * def_f * b2b_multiplier * dvp_multiplier * volume_factor
                    prob_over = (1 - poisson.cdf(p['granica'] - 0.5, adj_mu)) * 100

                    results.append({
                        "Igrač": p['ime'],
                        "Granica": p['granica'],
                        "Proj. Pts": round(adj_mu, 1),
                        "Over %": f"{round(prob_over, 1)}%",
                        "Confidence": f"{confidence}%",
                        "Volume": "⬆️ Raste" if volume_factor > 1.05 else ("⬇️ Pada" if volume_factor < 0.95 else "➡️ Stabilan"),
                        "Preporuka": "✅ OVER" if prob_over > 65 and confidence > 60 else ("❌ UNDER" if prob_over < 35 and confidence > 60 else "⚠️ RISKANTNO")
                    })
                except Exception as e:
                    st.error(f"Greška kod {p['ime']}: {e}")

        if results:
            res_df = pd.DataFrame(results)
            st.table(res_df)
            
            st.markdown("""
            **Upute za čitanje rezultata:**
            - **Volume:** Govori ti uzima li igrač više šuteva u zadnje vrijeme nego inače.
            - **Confidence:** Ako je ispod 50%, igrač previše oscilira (jednu zabije 5, drugu 25) i statistika mu nije pouzdana.
            - **Preporuka:** Program preporučuje par samo ako su i vjerojatnost i konstantnost (Confidence) visoki.
            """)
