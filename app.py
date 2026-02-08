import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import poisson
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats

# --- 1. POMOĆNE FUNKCIJE ---
def check_b2b(log):
    """Provjerava je li igrač igrao jučer na temelju njegovog loga."""
    try:
        if log.empty: return False
        last_game_str = log.iloc[0]['GAME_DATE']
        # Format datuma u NBA API je 'FEB 07, 2024'
        last_game_date = datetime.strptime(last_game_str, '%b %d, %Y').date()
        yesterday = datetime.now().date() - timedelta(days=1)
        return last_game_date == yesterday
    except:
        return False

def get_position_factor(opponent, position):
    """Vraća faktor težine na temelju pozicije i protivnika (DvP)."""
    # Primjer elite obrane protiv određenih pozicija
    elite_vs_centers = ["Timberwolves", "Celtics", "Heat", "76ers"]
    bad_vs_centers = ["Wizards", "Pistons", "Hornets", "Spurs"]
    
    if position == "C":
        if any(team in opponent for team in elite_vs_centers): return 0.88
        if any(team in opponent for team in bad_vs_centers): return 1.12
    
    # Možeš dodati i za PG (npr. protiv Jrue Holiday-a ili Derrick White-a)
    elite_vs_guards = ["Celtics", "Thunder", "Magic"]
    if position == "PG" or position == "SG":
        if any(team in opponent for team in elite_vs_guards): return 0.90
        
    return 1.0

# --- 2. UI POSTAVKE ---
st.set_page_config(page_title="NBA Pro Analytics", layout="wide")
st.title("🏀 NBA Pro Analytics (B2B & DvP Mode)")

nba_teams = sorted([team['nickname'] for team in teams.get_teams()])
positions = ["PG", "SG", "SF", "PF", "C"]

if 'batch_list' not in st.session_state:
    st.session_state.batch_list = []

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Dodaj Igrača")
    ime = st.text_input("Ime igrača", "Joel Embiid")
    pozicija = st.selectbox("Pozicija", positions)
    moj_tim = st.selectbox("Njegov tim", nba_teams)
    protivnik = st.selectbox("Protivnik", nba_teams)
    granica = st.number_input("Granica", value=25.5)
    
    if st.button("➕ Dodaj na listu"):
        st.session_state.batch_list.append({
            "ime": ime, "poz": pozicija, "tim": moj_tim, "protivnik": protivnik, "granica": granica
        })
        st.success(f"Dodan {ime}!")

    if st.button("🗑️ Očisti listu"):
        st.session_state.batch_list = []
        st.rerun()

# --- 4. ANALIZA ---
if st.session_state.batch_list:
    st.subheader("📋 Lista za obradu")
    st.write(pd.DataFrame(st.session_state.batch_list))
    
    if st.button("🚀 POKRENI ANALIZU"):
        results = []
        with st.spinner('Dohvaćam napredne podatke...'):
            stats_df = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced').get_data_frames()[0]
            l_pace = stats_df['PACE'].mean()
            l_def = stats_df['DEF_RATING'].mean()

            for p in st.session_state.batch_list:
                try:
                    p_search = players.find_players_by_full_name(p['ime'])
                    if not p_search: continue
                    
                    log = playergamelog.PlayerGameLog(player_id=p_search[0]['id'], season='2024-25').get_data_frames()[0]
                    
                    # Provjera B2B faktora
                    is_b2b = check_b2b(log)
                    b2b_multiplier = 0.90 if is_b2b else 1.0
                    
                    # DvP faktor
                    dvp_multiplier = get_position_factor(p['protivnik'], p['poz'])
                    
                    # Efikasnost (PTS po minuti)
                    eff = (log.head(10)['PTS'] / log.head(10)['MIN'].replace(0, 1)).mean()
                    proj_min = log['MIN'].mean()
                    
                    # Dohvaćanje timskog tempa i obrane
                    t_stats = stats_df[stats_df['TEAM_NAME'].str.contains(p['tim'])].iloc[0]
                    o_stats = stats_df[stats_df['TEAM_NAME'].str.contains(p['protivnik'])].iloc[0]
                    
                    pace_f = ((t_stats['PACE'] * o_stats['PACE']) / l_pace) / l_pace
                    def_f = o_stats['DEF_RATING'] / l_def

                    # Finalna kalkulacija
                    adj_mu = proj_min * eff * pace_f * def_f * b2b_multiplier * dvp_multiplier
                    prob_over = (1 - poisson.cdf(p['granica'] - 0.5, adj_mu)) * 100

                    results.append({
                        "Igrač": p['ime'],
                        "B2B": "DA ⚠️" if is_b2b else "NE",
                        "DvP": f"{dvp_multiplier}x",
                        "Proj. Pts": round(adj_mu, 1),
                        "Over %": round(prob_over, 1),
                        "Preporuka": "🔥 OVER" if prob_over > 65 else ("❄️ UNDER" if prob_over < 35 else "⚖️ NO BET")
                    })
                except Exception as e:
                    st.error(f"Greška kod {p['ime']}: {e}")

        if results:
            st.table(pd.DataFrame(results))
