import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy.stats import poisson
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats

# --- POMOĆNE FUNKCIJE ---
def get_injury_report():
    url = "https://www.rotowire.com/basketball/injury-report.php"
    try:
        r = requests.get(url, timeout=3)
        soup = BeautifulSoup(r.text, 'html.parser')
        injuries = {}
        for row in soup.find_all('tr', class_='injury-report__row'):
            team = row.find('td', class_='injury-report__team').text.strip()
            player = row.find('a').text.strip()
            status = row.find('td', class_='injury-report__status').text.strip()
            if team not in injuries: injuries[team] = []
            injuries[team].append({'name': player, 'status': status})
        return injuries
    except: return {}

# --- UI POSTAVKE ---
st.set_page_config(page_title="NBA Batch Analyzer", layout="wide")
st.title("🚀 NBA Batch Prop Analyzer")
st.markdown("Unesite više igrača odjednom i pronađite najbolje 'value' opklade za večeras.")

# --- INPUT SEKCIJA ---
with st.sidebar:
    st.header("Postavke Skupne Analize")
    st.info("Format: Ime Igrača, Tim, Protivnik, Granica")
    input_data = st.text_area(
        "Unesite listu (svaki igrač u novi red):",
        "Amen Thompson, Rockets, Lakers, 13.5\nJayson Tatum, Celtics, Knicks, 26.5\nLuka Doncic, Mavericks, Suns, 32.5"
    )
    spread_global = st.number_input("Prosječni Spread utakmica", value=5.5)
    run_batch = st.button("POKRENI SKUPNU ANALIZU")

# --- LOGIKA ANALIZE ---
if run_batch:
    results = []
    injury_data = get_injury_report()
    stats_df = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced').get_data_frames()[0]
    l_pace = stats_df['PACE'].mean()
    l_def = stats_df['DEF_RATING'].mean()

    lines = input_data.split('\n')
    
    progress_bar = st.progress(0)
    for i, line in enumerate(lines):
        try:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 4: continue
            
            ime, tim, protivnik, granica = parts[0], parts[1], parts[2], float(parts[3])
            
            # 1. Igrač & Efikasnost
            p_info = players.find_players_by_full_name(ime)[0]
            log = playergamelog.PlayerGameLog(player_id=p_info['id'], season='2024-25').get_data_frames()[0]
            log['PTS_PER_MIN'] = log['PTS'] / log['MIN'].replace(0, 1)
            eff = log.head(10)['PTS_PER_MIN'].mean()
            proj_min = log['MIN'].mean()
            
            # 2. Timski faktori
            t_pace = stats_df[stats_df['TEAM_NAME'].str.contains(tim, case=False)]['PACE'].values[0]
            o_pace = stats_df[stats_df['TEAM_NAME'].str.contains(protivnik, case=False)]['PACE'].values[0]
            o_def = stats_df[stats_df['TEAM_NAME'].str.contains(protivnik, case=False)]['DEF_RATING'].values[0]
            
            pace_f = ((t_pace * o_pace) / l_pace) / l_pace
            def_f = o_def / l_def
            
            # 3. Ozljede
            usage = 1.0
            for t_name, p_list in injury_data.items():
                if tim.lower() in t_name.lower():
                    usage += sum(0.05 for p in p_list if "Out" in p['status'])

            # 4. Kalkulacija
            adj_mu = proj_min * eff * pace_f * def_f * usage
            prob_over = (1 - poisson.cdf(granica - 0.5, adj_mu)) * 100
            
            results.append({
                "Igrač": ime,
                "Granica": granica,
                "Proj. Poeni": round(adj_mu, 1),
                "Over %": round(prob_over, 1),
                "Under %": round(100 - prob_over, 1),
                "Status": "🔥 OVER" if prob_over > 65 else ("❄️ UNDER" if prob_over < 35 else "⚖️ NO BET")
            })
        except Exception as e:
            st.error(f"Greška kod igrača {line}: {e}")
        
        progress_bar.progress((i + 1) / len(lines))

    # --- PRIKAZ REZULTATA ---
    if results:
        final_df = pd.DataFrame(results).sort_values(by="Over %", ascending=False)
        
        st.subheader("📋 Pregled svih parova")
        st.table(final_df)
        
        # Vizualizacija najboljih šansi
        st.subheader("📊 Usporedba Vjerojatnosti (Over)")
        st.bar_chart(final_df.set_index('Igrač')['Over %'])
        
        

        st.success("Analiza završena! Parovi s najvišim 'Over %' su statistički najizgledniji.")

