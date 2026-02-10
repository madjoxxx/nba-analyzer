import numpy as np


# =========================
# ENSEMBLE PREDICTION
# =========================

def ensemble_prediction(ml_pred, base_pred):

    # weighted ensemble
    return ml_pred * 0.7 + base_pred * 0.3


# =========================
# VOLATILITY
# =========================

def volatility_flag(std):

    if std < 4:
        return "LOW"
    elif std < 7:
        return "MED"
    else:
        return "HIGH"


# =========================
# MULTI LINE EDGE CURVE
# =========================

def line_sensitivity(pred, std, center_line):

    lines = [
        center_line - 2,
        center_line - 1,
        center_line,
        center_line + 1,
        center_line + 2,
    ]

    results = []

    for line in lines:

        sims = np.random.normal(pred, std, 8000)
        over = np.mean(sims > line)

        edge = pred - line

        results.append({
            "line": round(line,1),
            "over": round(over*100,1),
            "edge": round(edge,2)
        })

    return results


# =========================
# BET SIGNAL
# =========================

def bet_signal(edge, over_prob, confidence, consistency):

    score = (
        edge * 10 +
        over_prob * 40 +
        confidence * 0.3 +
        consistency * 0.2
    )

    if score > 95:
        return "STRONG BET"
    elif score > 75:
        return "BET"
    elif score > 60:
        return "LEAN"
    else:
        return "PASS"


# =========================
# STAKE SUGGESTION
# =========================

def stake_size(confidence, edge):

    raw = (confidence/100) * max(edge,0)

    pct = min(raw * 2, 5)

    return round(pct,2)
