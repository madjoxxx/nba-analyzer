import numpy as np
import pandas as pd

ML_AVAILABLE = True
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
except:
    ML_AVAILABLE = False


# =====================
# FEATURES
# =====================

def build_features(df):

    df = df.sort_values("GAME_DATE")

    df["PTS_r5"] = df["PTS"].rolling(5).mean()
    df["PTS_r10"] = df["PTS"].rolling(10).mean()

    df["MIN_r5"] = df["MIN"].rolling(5).mean()
    df["MIN_r10"] = df["MIN"].rolling(10).mean()

    df["USAGE"] = df["FGA"] + df["FTA"] * 0.44

    df["HOME"] = df["MATCHUP"].str.contains("vs").astype(int)

    df = df.dropna()

    feats = [
        "PTS_r5","PTS_r10",
        "MIN_r5","MIN_r10",
        "USAGE","HOME"
    ]

    return df[feats], df["PTS"]


# =====================
# BASELINE
# =====================

def baseline(df):
    last5 = df["PTS"].tail(5)
    return last5.mean(), last5.std()


# =====================
# MINUTES MODEL
# =====================

def predict_minutes(df):

    last5 = df["MIN"].tail(5)
    trend = last5.mean()

    if last5.iloc[-1] > trend:
        trend *= 1.03

    return trend


# =====================
# CONSISTENCY
# =====================

def consistency_score(df):

    std = df["PTS"].tail(10).std()

    if std < 4:
        return 90
    if std < 6:
        return 75
    if std < 9:
        return 60
    return 45


# =====================
# ML TRAIN
# =====================

def train_ml(X, y):

    if not ML_AVAILABLE:
        return None, y.std()

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        random_state=42
    )

    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    mae = np.mean(np.abs(pred - yte))

    return model, mae


# =====================
# MONTE CARLO
# =====================

def monte(mean, std, line):

    sims = np.random.normal(mean, std, 15000)
    return np.mean(sims > line)


# =====================
# BACKTEST
# =====================

def backtest_hit_rate(df, line):

    hits = 0
    tests = 0

    if len(df) < 15:
        return 0

    for i in range(12, len(df)):

        train_window = df.iloc[:i]
        test_game = df.iloc[i]

        mean = train_window["PTS"].tail(5).mean()

        if test_game["PTS"] > line:
            hits += 1

        tests += 1

    if tests == 0:
        return 0

    return hits / tests
