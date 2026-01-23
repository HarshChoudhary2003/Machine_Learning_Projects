import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_synthetic_data(n_samples: int = 8000) -> pd.DataFrame:
    battery_percent = np.random.uniform(5, 100, n_samples)
    time_since_unplugged_min = np.random.uniform(1, 300, n_samples)
    cpu_usage = np.random.uniform(5, 95, n_samples)
    gpu_usage = np.random.uniform(0, 90, n_samples)
    screen_brightness = np.random.uniform(10, 100, n_samples)
    num_apps_running = np.random.randint(3, 40, n_samples)
    heavy_app_running = np.random.binomial(1, 0.3, n_samples)
    wifi_on = np.random.binomial(1, 0.8, n_samples)
    bluetooth_on = np.random.binomial(1, 0.4, n_samples)
    power_mode = np.random.choice(
        ["battery_saver", "balanced", "performance"],
        size=n_samples,
        p=[0.3, 0.5, 0.2]
    )
    fan_speed_rpm = np.random.uniform(1200, 4500, n_samples)
    device_temperature = np.random.uniform(40, 95, n_samples)

    base_drain = 0.02

    drain_cpu = cpu_usage * 0.00025
    drain_gpu = gpu_usage * 0.0002
    drain_brightness = screen_brightness * 0.00015
    drain_heavy = heavy_app_running * 0.03
    drain_power_mode = np.where(
        power_mode == "battery_saver", -0.01,
        np.where(power_mode == "balanced", 0.0, 0.02)
    )
    drain_temp = np.clip((device_temperature - 60) * 0.0005, 0, 0.04)

    drain_per_min = base_drain + drain_cpu + drain_gpu + drain_brightness \
                    + drain_heavy + drain_power_mode + drain_temp

    drain_per_min = np.clip(drain_per_min, 0.005, 0.15)

    remaining_minutes = battery_percent / (drain_per_min * 100)
    remaining_minutes = np.clip(remaining_minutes, 5, 600)

    df = pd.DataFrame({
        "battery_percent": battery_percent,
        "time_since_unplugged_min": time_since_unplugged_min,
        "cpu_usage": cpu_usage,
        "gpu_usage": gpu_usage,
        "screen_brightness": screen_brightness,
        "num_apps_running": num_apps_running,
        "heavy_app_running": heavy_app_running,
        "wifi_on": wifi_on,
        "bluetooth_on": bluetooth_on,
        "power_mode": power_mode,
        "fan_speed_rpm": fan_speed_rpm,
        "device_temperature": device_temperature,
        "remaining_minutes": remaining_minutes
    })

    return df


if __name__ == "__main__":
    print("Generating dataset...")

    # Always save relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(project_root, "data")
    os.makedirs(data_folder, exist_ok=True)

    output_file = os.path.join(data_folder, "synthetic_battery_data.csv")

    df = generate_synthetic_data()
    df.to_csv(output_file, index=False)

    print("Dataset shape:", df.shape)
    print(f"Saved successfully → {output_file}")
