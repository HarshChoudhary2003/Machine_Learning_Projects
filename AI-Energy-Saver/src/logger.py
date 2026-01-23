import os
import pandas as pd
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "data", "battery_log.csv")

def log_snapshot(features: dict, predicted_minutes: float):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(),
        "battery_percent": features["battery_percent"],
        "cpu_usage": features["cpu_usage"],
        "screen_brightness": features["screen_brightness"],
        "num_apps_running": features["num_apps_running"],
        "power_mode": features["power_mode"],
        "device_temperature": features.get("device_temperature", None),
        "predicted_minutes": predicted_minutes,
    }

    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(LOG_PATH, index=False)


def load_history():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return None
