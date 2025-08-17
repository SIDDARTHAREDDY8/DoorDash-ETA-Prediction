import os, requests
from typing import Optional, Dict

def kelvin_to_c(tk: float) -> Optional[float]:
    return tk - 273.15 if tk is not None else None

def get_weather_openweather(lat: float, lon: float) -> Dict:
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {"temperature_c": None, "weather": None, "precip": None, "wind_mps": None}
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key}
    r = requests.get(url, params=params, timeout=15); r.raise_for_status()
    data = r.json()
    temp_c = kelvin_to_c(data.get("main", {}).get("temp"))
    weather_main = data["weather"][0]["main"] if data.get("weather") else None
    rain = (data.get("rain", {}) or {}).get("1h", 0) or 0
    snow = (data.get("snow", {}) or {}).get("1h", 0) or 0
    precip = 1 if (rain > 0 or snow > 0) else 0
    wind = (data.get("wind", {}) or {}).get("speed")
    return {"temperature_c": temp_c, "weather": weather_main, "precip": precip, "wind_mps": wind}

def get_traffic_congestion(lat: float, lon: float) -> Dict:
    # Placeholder for TomTom/Google APIs; returns a simulated index
    from datetime import datetime
    h = datetime.utcnow().hour
    if 7 <= h <= 9 or 16 <= h <= 19: idx = 7.5
    elif 11 <= h <= 14: idx = 5.0
    else: idx = 2.0
    return {"congestion_index": idx}