# agent.py
import time
import psutil
import requests
import socket

SERVER_URL = "http://127.0.0.1:8000/ingest"  # change to your deployed API URL
DEVICE_ID = socket.gethostname()             # or set manually like "rohan-laptop"


def get_stats():
    try:
        battery = psutil.sensors_battery()
        battery_percent = battery.percent if battery else None
    except Exception:
        battery_percent = None

    cpu_usage = psutil.cpu_percent(interval=0.5)
    ram_usage = psutil.virtual_memory().percent
    num_apps_running = len(psutil.pids())

    # temperature (optional, many laptops won't expose this => safe fallback)
    device_temperature = None
    try:
        # some systems may have temperature sensors
        temps = psutil.sensors_temperatures()
        if temps:
            # pick first sensor's first reading
            sensor = list(temps.values())[0][0]
            device_temperature = sensor.current
    except Exception:
        device_temperature = None

    return {
        "device_id": DEVICE_ID,
        "battery_percent": battery_percent,
        "cpu_usage": cpu_usage,
        "ram_usage": ram_usage,
        "num_apps_running": num_apps_running,
        "device_temperature": device_temperature,
    }


def main():
    print(f"Starting agent for device_id={DEVICE_ID}")
    while True:
        stats = get_stats()
        try:
            res = requests.post(SERVER_URL, json=stats, timeout=5)
            if res.status_code != 200:
                print("Send error:", res.text)
        except Exception as e:
            print("Connection error:", e)
        time.sleep(5)  # send every 5 seconds


if __name__ == "__main__":
    main()
