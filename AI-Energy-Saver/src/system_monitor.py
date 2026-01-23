import psutil

# If temperature access via WMI fails, we handle it safely
try:
    import wmi
    w = wmi.WMI()
    HAS_WMI = True
except:
    HAS_WMI = False


def get_system_stats():
    # Battery
    try:
        battery = psutil.sensors_battery()
        battery_percent = battery.percent if battery else None
        power_plugged = battery.power_plugged if battery else None
    except:
        battery_percent = None
        power_plugged = None

    # CPU
    try:
        cpu_usage = psutil.cpu_percent(interval=0.5)
    except:
        cpu_usage = 0

    # RAM
    try:
        ram_usage = psutil.virtual_memory().percent
    except:
        ram_usage = 0

    # Count running apps
    try:
        num_apps_running = len(psutil.pids())
    except:
        num_apps_running = 0

    # Temperature
    temperature = None
    if HAS_WMI:
        try:
            sensors = w.MSAcpi_ThermalZoneTemperature()
            if sensors:
                temperature = (sensors[0].CurrentTemperature / 10) - 273.15
        except:
            temperature = None

    # Even if temperature not available, return safe object
    return {
        "battery_percent": battery_percent,
        "power_plugged": power_plugged,
        "cpu_usage": cpu_usage,
        "ram_usage": ram_usage,
        "num_apps_running": num_apps_running,
        "device_temperature": round(temperature, 1) if temperature else None,
    }
