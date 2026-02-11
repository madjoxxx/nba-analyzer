import numpy as np
import pandas as pd

from nba_api.stats.endpoints import playergamelog
from sklearn.linear_model import LinearRegression


# -------------------------
# LOAD GAMES
# -------------------------

def get_games(pid):

    try:
        df = playergamelog.PlayerGameLog(
            player_id=pid,
            season="2024-25"
        ).get_data_frames()[0]

        df = df.sort_values("GAME_DATE")

        if len(df) < 15:
            return None

        return df

    except:
        return None


# -------------------------
# OPPONENT PARSE
# -------------------------

def get_opponent(row):
    # MATCHUP format: "LAL vs BOS" or "LAL @ BOS"
    s = row["MATCHUP"]
    return s.split()[-1]


# -------------------------
# MATCHUP ENGINE
# -------------------------

def opponent_features(df):

    df = df.copy()
    df["OPP"] = df.apply(get_opponent, axis=1)

    season_avg = df["PTS"].mean()

    opp_table = df.groupby("OPP").agg({
        "PTS":"mean",
        "MIN":"mean",
        "FGA":"mean",
        "GAME_ID":"count"
    }).reset_index()

    opp_table["pts_diff"] = opp_table["PTS"] - season_avg

    return opp_table, season_avg


def opponent_adjustment(df):

    opp_table, season_avg = opponent_features(df)

    # recent opponents weight
    recent = df.tail(5).copy()
    recent["OPP"] = recent.apply(get_opponent, axis=1)

    adj = 0
    weight = 0

    for opp in recent["OPP"]:

        row = opp_table[opp_table["OPP"] == opp]

        if len(row) == 0:
            continue

        diff = float(row["pts_diff"].iloc[0])
        games = int(row["GAME_ID"].iloc[0])

        sample_weight = min(games / 5, 1)

        adj += diff * sample_weight
        weight += sample_weight

    if weight == 0:
        return 0

    return adj / weight


# -------------------------
# LINEUP FEATURES
# -------------------------

def lineup_features(df):

    f = {}

    min_l5 = df["MIN"].tail(5).mean()
    min_l15 = df["MIN"].tail(15).mean()

    fga_l5 = df["FGA"].tail(5).mean()
    fga_l15 = df["FGA"].tail(15).mean()

    f["minutes_spike"] = min_l5 - min_l15
    f["usage_spike"] = fga_l5 - fga_l15

    f["starter_flag"] = 1 if min_l5 >= 30 else 0
    f["rotation_risk"] = df["MIN"].tail(10).std()

    return f


# -------------------------
# FEATURE ENGINE
# -------------------------

def build_features(df):

    f = {}

    f["pts_l5"] = df["PTS"].tail(5).mean()
    f["pts_l10"] = df["PTS"].tail(10).mean()
    f["pts_season"] = df["PTS"].mean()

    f["min_l5"] = df["MIN"].tail(5).mean()
    f["fga_l5"] = df["FGA"].tail(5).mean()
    f["fta_l5"] = df["FTA"].tail(5).mean()

    f["shot_volume"] = f["fga_l5"] + f["fta_l5"] * 0.44
    f["form_delta"] = f["pts_l5"] - f["pts_season"]

    f["std_pts"] = df["PTS"].tail(10).std()

    f.update(lineup_features(df))

    f["opp_adj"] = opponent_adjustment(df)

    return f


# -------------------------
# ML PROJECTION
# -------------------------

def project(df):

    X = df[["MIN","FGA","FTA"]].tail(20)
    y = df["PTS"].tail(20)

    if len(X) < 10:
        return None, None

    model = LinearRegression()
    model.fit(X, y)

    feats = build_features(df)

    pred = model.predict([[
        feats["min_l5"],
        feats["fga_l5"],
        feats["fta_l5"]
    ]])[0]

    # -------------------------
    # ADJUSTMENTS STACK
    # -------------------------

    pred += feats["form_delta"] * 0.4
    pred += feats["minutes_spike"] * 0.35
    pred += feats["usage_spike"] * 0.6

    # matchup adjustment
    pred += feats["opp_adj"] * 0.7

    if feats["starter_flag"]:
        pred *= 1.05

    if feats["rotation_risk"] > 8:
        pred *= 0.94

    return float(pred), feats


# -------------------------
# PROBABILITY
# -------------------------

def prob_over(pred, line, feats):

    sigma = max(feats["std_pts"], 4)

    z = (pred - line) / sigma

    return float(1 / (1 + np.exp(-z)))


# -------------------------
# BACKTEST
# -------------------------

def backtest(df, line):

    w = df.tail(12)

    if len(w) == 0:
        return 0.5

    return (w["PTS"] > line).mean()


# -------------------------
# CONFIDENCE
# -------------------------

def confidence(p, hit):

    return round(p*60 + hit*40, 1)


# -------------------------
# VOLATILITY
# -------------------------

def volatility(df):

    s = df["PTS"].tail(10).std()

    if s < 5: return "LOW"
    if s < 9: return "MED"
    return "HIGH"


def consistency(df):

    m = df["PTS"].mean()
    s = df["PTS"].std()

    if m == 0:
        return 50

    return round((1 - s/m)*100,1)


# -------------------------
# STAKE
# -------------------------

def stake(edge, conf):

    if conf > 70 and abs(edge) > 3:
        return 5
    if conf > 65:
        return 4
    if conf > 58:
        return 3
    return 2
