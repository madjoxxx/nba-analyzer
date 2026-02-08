import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy.stats import poisson
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats

# --- 1. FUNKCIJA ZA AUTOMATSKE OZLJEDE ---
def get_injury_report():
    url = "https://www.rotowire.com/basketball/injury-report.php"
    try:
        r = requests.get(url, timeout=5)
        soup = BeautifulSoup(r.text, 'html.parser')
        injuries = {}
        for row in soup.find_all('tr', class_='injury-report__row'):
            team_elem = row.find('td', class_='injury-report__team')
            player_elem = row.find('a')
            status_elem = row.find('td', class_='injury-report__status')
            if team_elem and player_elem and status_elem:
                team = team_elem.text.strip()
                player = player_elem.text.strip()
                status = status_elem.text.strip()
                if team not in injuries: injuries[team] = []
                injuries[team].append({'name': player, 'status': status})
        return injuries
    except:
        return {}

# --- 2. PODACI I UI POSTAVKE ---
nba_teams = sorted([team['nickname'] for team in teams.get_teams()])
st.set_page_config(page_title="NBA AI Predictor", layout="wide")
st.title("🏀 NBA Ultimate Predictor (Vrhunska Analiza)")

if 'batch_list' not in st.session_state:
    st.session_state.batch_list = []

# --- 3. SIDEBAR ZA DODAVANJE IGRAČA ---
with st.sidebar:
    st.header("📍 Dodaj par za listić")
    ime = st.text_input("Ime i prezime igrača", "Jayson Tatum")
    moj_tim = st.selectbox("Njegov tim", nba_teams)
    protivnik = st.selectbox("Protivnik", nba_teams)
    granica = st.number_input("Granica poena", value=25.5, step=0.5)
    
    if st.button("➕ Dodaj na listu"):
        st.session_state.batch_list.append({
            "ime": ime, "tim": moj_tim, "protivnik": protivnik, "granica": granica
        })
        st.success(f"Dodan {ime}!")

    if st.button("🗑️ Očisti sve"):
        st.session_state.batch_list = []
        st.rerun()

# --- 4. GLAVNA ANALIZA ---
if st.session_state.batch_list:
    st.subheader("📋 Trenutna lista za analizu")
    st.write(pd.DataFrame(st.session_state.batch_list))
    
    if st.button("🚀 POKRENI ANALIZU (Uključi Ozljede i Tempo)"):
        injury_data = get_injury_report()
        results = []
        
        with st.spinner('Skeniram NBA bazu i ozljede...'):
            stats_df = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced').get_data_frames()[0]
            l_pace = stats_df['PACE'].mean()
            l_def = stats_df['DEF_RATING'].mean()

            for p in st.session_state.batch_list:
                try:
                    # Statistika igrača (Efikasnost)
                    p_search = players.find_players_by_full_name(p['ime'])
                    if not p_search: continue
                    
                    log = playergamelog.PlayerGameLog(player_id=p_search[0]['id'], season='2024-25').get_data_frames()[0]
                    log['PTS_PER_MIN'] = log['PTS'] / log['MIN'].replace(0, 1)
                    eff = log.head(10)['PTS_PER_MIN'].mean()
                    proj_min = log['MIN'].mean()

                    # Tempo i Obrana
                    t_stats = stats_df[stats_df['TEAM_NAME'].str.contains(p['tim'])].iloc[0]
                    o_stats = stats_df[stats_df['TEAM_NAME'].str.contains(p['protivnik'])].iloc[0]
                    pace_f = ((t_stats['PACE'] * o_stats['PACE']) / l_pace) / l_pace
                    def_f = o_stats['DEF_RATING'] / l_def

                    # OZLJEDE (Usage Bump)
                    usage_bump = 1.0
                    missing_names = []
                    for t_name, p_list in injury_data.items():
                        if p['tim'].lower() in t_name.lower():
                            for out_p in p_list:
                                if "Out" in out_p['status']:
                                    usage_bump += 0.06 # +6% za svakog startera koji ne igra
                                    missing_names.append(out_p['name'])

                    # Konačna projekcija
                    adj_mu = proj_min * eff * pace_f * def_f * usage_bump
                    prob_over = (1 - poisson.cdf(p['granica'] - 0.5, adj_mu)) * 100

                    results.append({
                        "Igrač": p['ime'],
                        "Granica": p['granica'],
                        "Fali u timu": ", ".join(missing_names) if missing_names else "Nitko",
                        "Vjerojatnost %": round(prob_over, 1),
                        "Analiza": "🔥 OVER" if prob_over > 62 else ("❄️ UNDER" if prob_over < 38 else "⚖️ NO BET")
                    })
                except:
                    continue

        if results:
            st.table(pd.DataFrame(results))
            st.info("💡 Savjet: Fokusirajte se na parove s preko 65% šanse.")
else:
    st.info("Sidebar: Dodaj igrače koje želiš analizirati.")
