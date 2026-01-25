import requests
from langchain.tools import tool

@tool
def get_weather(latitude: float, longitude: float):
    """Get weather forecast using Open-Meteo API"""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        f"&daily=temperature_2m_max&timezone=auto"
    )
    response = requests.get(url)
    return response.json()
