from typing import Dict, List

def recommend_actions(features: Dict, predicted_minutes: float) -> List[str]:
    recs = []

    if features["screen_brightness"] > 70:
        recs.append("Reduce screen brightness below 60% to extend battery life.")

    if features["cpu_usage"] > 70 or features["heavy_app_running"] == 1:
        recs.append("Close heavy apps like games, video editors, or many browser tabs.")

    if features["power_mode"] == "performance":
        recs.append("Switch to Balanced or Battery Saver mode for better endurance.")

    if features["wifi_on"] == 1 and predicted_minutes < 60:
        recs.append("Turn off WiFi when internet is not needed.")

    if features["bluetooth_on"] == 1 and predicted_minutes < 90:
        recs.append("Turn off Bluetooth to reduce extra power consumption.")

    if features["device_temperature"] > 80:
        recs.append("Let the laptop cool down. High temperature speeds up battery drain.")

    if features["battery_percent"] < 20 and predicted_minutes < 30:
        recs.append("Consider plugging in the charger soon to avoid sudden shutdown.")

    if not recs:
        recs.append("Your current usage looks efficient. No urgent actions required.")

    return recs
