import streamlit as st
import pandas as pd
from scipy.stats import poisson
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats

# --- PODACI ---
nba_teams = sorted([team['nickname'] for team in teams.get_teams()])

# --- UI POSTAVKE ---
st.set_page_config(page_title="NBA No-Error Predictor", layout="wide")
st.title("🏀 NBA Prop Analyzer (Select Mode)")

# Inicijalizacija liste za batch ako ne postoji
if 'batch_list' not in st.session_state:
    st.session_state.batch_list = []

with st.sidebar:
    st.header("1. Dodaj Igrača")
    ime = st.text_input("Ime i prezime igrača", "LeBron James")
    moj_tim = st.selectbox("Njegov tim", nba_teams, index=nba_teams.index("Lakers"))
    protivnik = st.selectbox("Protivnik", nba_teams, index=nba_teams.index("Warriors"))
    granica = st.number_input("Granica poena", value=25.5, step=0.5)
    
    if st.button("➕ Dodaj na listu za analizu"):
        novo = {"ime": ime, "tim": moj_tim, "protivnik": protivnik, "granica": granica}
        st.session_state.batch_list.append(novo)
        st.success(f"Dodan {ime}!")

    if st.button("🗑️ Očisti listu"):
        st.session_state.batch_list = []
        st.rerun()

# --- GLAVNI EKRAN ---
st.subheader("📋 Lista za analizu")
if st.session_state.batch_list:
    st.write(pd.DataFrame(st.session_state.batch_list))
    
    if st.button("🚀 POKRENI ANALIZU SVIH PAROVA"):
        results = []
        with st.spinner('Analiziram...'):
            # Dohvati zajedničku statistiku jednom (štedi vrijeme)
            stats_df = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced').get_data_frames()[0]
            l_pace = stats_df['PACE'].mean()
            l_def = stats_df['DEF_RATING'].mean()

            for p in st.session_state.batch_list:
                try:
                    # Igrač
                    p_search = players.find_players_by_full_name(p['ime'])
                    if not p_search:
                        st.error(f"Igrač {p['ime']} nije pronađen.")
                        continue
                    
                    log = playergamelog.PlayerGameLog(player_id=p_search[0]['id'], season='2024-25').get_data_frames()[0]
                    log['PTS_PER_MIN'] = log['PTS'] / log['MIN'].replace(0, 1)
                    eff = log.head(10)['PTS_PER_MIN'].mean()
                    proj_min = log['MIN'].mean()

                    # Timovi (točna podudaranja iz liste)
                    t_stats = stats_df[stats_df['TEAM_NAME'].str.contains(p['tim'])].iloc[0]
                    o_stats = stats_df[stats_df['TEAM_NAME'].str.contains(p['protivnik'])].iloc[0]
                    
                    pace_f = ((t_stats['PACE'] * o_stats['PACE']) / l_pace) / l_pace
                    def_f = o_stats['DEF_RATING'] / l_def

                    adj_mu = proj_min * eff * pace_f * def_f
                    prob_over = (1 - poisson.cdf(p['granica'] - 0.5, adj_mu)) * 100

                    results.append({
                        "Igrač": p['ime'],
                        "Granica": p['granica'],
                        "Projekcija": round(adj_mu, 1),
                        "Vjerojatnost %": round(prob_over, 1),
                        "Tip": "🔥 OVER" if prob_over > 60 else "❄️ UNDER"
                    })
                except Exception as e:
                    st.error(f"Greška kod {p['ime']}: {e}")

        if results:
            st.table(pd.DataFrame(results))
else:
    st.info("Dodaj igrače u sidebaru kako bi započeo analizu.")
