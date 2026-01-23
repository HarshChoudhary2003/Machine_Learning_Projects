from typing import Dict, Tuple
import numpy as np
import pandas as pd

def compute_efficiency_score(features: Dict, predicted_minutes: float) -> Tuple[int, str]:
    """
    Returns:
        score: 0-100
        label: 'Good' / 'Moderate' / 'Poor'
    """

    score = 100

    # High CPU usage penalty
    if features["cpu_usage"] > 80:
        score -= 25
    elif features["cpu_usage"] > 60:
        score -= 15
    elif features["cpu_usage"] > 40:
        score -= 5

    # Brightness penalty
    if features["screen_brightness"] > 80:
        score -= 20
    elif features["screen_brightness"] > 60:
        score -= 10

    # Heavy app penalty
    if features["heavy_app_running"] == 1:
        score -= 15

    # Temperature penalty
    temp = features.get("device_temperature", 60)
    if temp > 85:
        score -= 20
    elif temp > 75:
        score -= 10

    # Power mode penalty
    if features["power_mode"] == "performance":
        score -= 15
    elif features["power_mode"] == "balanced":
        score -= 5

    # Very low predicted time penalty
    if predicted_minutes < 30:
        score -= 20
    elif predicted_minutes < 60:
        score -= 10

    score = int(np.clip(score, 0, 100))

    if score >= 80:
        label = "Good"
    elif score >= 40:
        label = "Moderate"
    else:
        label = "Poor"

    return score, label


def forecast_battery_curve(current_percent: float,
                           predicted_minutes: float,
                           horizon_minutes: int = 120,
                           step: int = 5) -> pd.DataFrame:
    """
    Simple forecast assuming constant drain rate based on current prediction.
    Returns a DataFrame with 'minute' and 'battery_percent'.
    """
    if predicted_minutes <= 0 or current_percent <= 0:
        return pd.DataFrame({"minute": [], "battery_percent": []})

    drain_per_min = current_percent / predicted_minutes  # % per minute

    minutes = list(range(0, horizon_minutes + 1, step))
    percents = []

    for m in minutes:
        p = current_percent - drain_per_min * m
        p = max(p, 0)
        percents.append(p)

    return pd.DataFrame({"minute": minutes, "battery_percent": percents})
