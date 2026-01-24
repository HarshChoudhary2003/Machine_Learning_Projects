# AI Energy Saver 🔋

An intelligent system monitoring and energy optimization tool that tracks device telemetry and provides recommendations for energy saving.

## Features
- **Real-time Telemetry:** Tracks CPU usage, RAM usage, Battery levels, and Device temperature.
- **Remote Monitoring:** FastAPI-based server to ingest data from multiple devices.
- **Streamlit Dashboard:** Visualizes energy consumption and system health.
- **Smart Recommendations:** Provides actionable insights to reduce energy footprint.

## Project Structure
- `agent.py`: Client-side script to collect and send system stats.
- `remote_server.py`: FastAPI server for data ingestion.
- `src/streamlit_app.py`: Web dashboard for monitoring.
- `src/system_monitor.py`: Core logic for system metric collection.

## How to Run
1. **Service Side:**
   ```bash
   uvicorn remote_server:app --reload
   ```
2. **Agent Side:**
   ```bash
   python agent.py
   ```
3. **Dashboard:**
   ```bash
   streamlit run src/streamlit_app.py
   ```

## Requirements
- Python 3.x
- `psutil`, `fastapi`, `streamlit`, `requests`, `uvicorn`
