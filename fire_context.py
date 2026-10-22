"""Real-world context for one fire location: place name, weather and fire-weather risk.

Used by the map's fire information panel (src/dashboard.py -> /api/fire-context).

Nothing here is made up. Every value comes from:
  - Open-Meteo (https://open-meteo.com): weather forecast + the past week's rain
  - OpenStreetMap Nominatim (https://nominatim.org): the place name
and the risk level is the Fosberg Fire Weather Index worked out from that weather.
SkyAlert's own prediction model isn't built yet, so there is no wildfire
probability: the API says so instead of inventing a number.
"""
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

# EXTERNAL API: Open-Meteo. Free, no key. Gives current weather, the hourly forecast and the past week's rain for a lat/lon (used for the weather tiles and the risk level)
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
# EXTERNAL API: OpenStreetMap Nominatim. Free, no key, max 1 request per second. Turns a lat/lon into a place name like "Poinsett County, Arkansas"
GEOCODE_URL ="https://nominatim.openstreetmap.org/reverse"
# Nominatim's usage policy asks every app to identify itself
USER_AGENT = "SkyAlert/0.1 (student wildfire project; https://github.com/aisutd/skyalert)"

FORECAST_HOURS = 24      # the forecast window the risk level covers
RAIN_LOOKBACK_DAYS = 7   # "recent rainfall" = total over this many days
CACHE_SECONDS = 30 * 60  # weather changes slowly; reuse answers for 30 minutes

# Fosberg index (0-100) -> risk level. 50+ is the level fire forecasters usually
# treat as critical fire weather; the lower bands are SkyAlert's own split.
RISK_LEVELS = [
    (50, "critical", "Critical"),
    (40, "high", "High"),
    (25, "moderate", "Moderate"),
    (0, "low", "Low"),
]

_cache = {}
_cache_lock = threading.Lock()
# Nominatim allows at most 1 request per second, so requests take turns
_geocode_lock = threading.Lock()
_last_geocode = 0.0


def fosberg_index(temp_f, humidity, wind_mph):
    """Fosberg Fire Weather Index (Fosberg 1978): how much the weather favours fire spread.

    Combines how dry dead vegetation gets (from temperature + humidity) with wind
    speed. 0 = no fire-weather concern, 100 = the most extreme.
    """
    h, t = humidity, temp_f
    # Equilibrium moisture content of dead fuel (%), formula depends on humidity
    if h < 10:
        m = 0.03229 + 0.281073 * h - 0.000578 * h * t
    elif h <= 50:
        m = 2.22749 + 0.160107 * h - 0.01478 * t
    else:
        m = 21.0606 + 0.005565 * h ** 2 - 0.00035 * h * t - 0.483199 * h
    x = m / 30
    moisture_damping = 1 - 2 * x + 1.5 * x ** 2 - 0.5 * x ** 3
    ffwi = moisture_damping * math.sqrt(1 + wind_mph ** 2) / 0.3002
    return max(0.0, min(100.0, ffwi))


def risk_level(ffwi):
    for threshold, key, label in RISK_LEVELS:
        if ffwi >= threshold:
            return {"key": key, "label": label}


def reverse_geocode(lat, lon):
    """Turn coordinates into a place name, e.g. "Gunnison County, Colorado"."""
    global _last_geocode
    with _geocode_lock:
        wait = 1.0 - (time.time() - _last_geocode)
        if wait > 0:
            time.sleep(wait)
        _last_geocode = time.time()
        response = requests.get(
            GEOCODE_URL,
            params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10, "addressdetails": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
    response.raise_for_status()
    address = response.json().get("address", {})
    place = address.get("city") or address.get("town") or address.get("village") or address.get("hamlet")
    parts = [p for p in (place, address.get("county"), address.get("state")) if p]
    return {
        "name": ", ".join(parts) or address.get("country") or "Unknown area",
        "country": address.get("country"),
    }


def fetch_weather(lat, lon):
    """Current weather, the next 24 h of hourly forecast, and the past week's rain."""
    response = requests.get(
        WEATHER_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_gusts_10m,"
                      "vapour_pressure_deficit,soil_moisture_0_to_1cm",
            "daily": "precipitation_sum",
            "past_days": RAIN_LOOKBACK_DAYS,
            "forecast_days": 2,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "auto",  # times come back in the fire's local time zone
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def summarize_weather(raw):
    """Pick out what the panel shows and work out the fire-weather risk."""
    current = raw["current"]
    hourly = raw["hourly"]

    # Hourly data starts a week ago; find the current hour, then take the next 24
    current_hour = current["time"][:13]  # "2026-10-14T20"
    start = next(i for i, t in enumerate(hourly["time"]) if t[:13] >= current_hour)
    window = range(start, min(start + FORECAST_HOURS, len(hourly["time"])))

    # Fire-weather index for every hour in the window; the risk is set by the worst hour
    hours = []
    for i in window:
        temp, rh, wind = hourly["temperature_2m"][i], hourly["relative_humidity_2m"][i], hourly["wind_speed_10m"][i]
        if None in (temp, rh, wind):
            continue
        hours.append({
            "time": hourly["time"][i], "temp": temp, "humidity": rh, "wind": wind,
            "gust": hourly["wind_gusts_10m"][i], "ffwi": fosberg_index(temp, rh, wind),
        })
    peak = max(hours, key=lambda h: h["ffwi"])

    # Rain over the last 7 full days (today is left out: it's partly forecast)
    today = current["time"][:10]
    past_days = [
        (day, rain) for day, rain in zip(raw["daily"]["time"], raw["daily"]["precipitation_sum"])
        if day < today and rain is not None
    ][-RAIN_LOOKBACK_DAYS:]
    rain_total = sum(rain for _, rain in past_days)

    return {
        "timezone": raw.get("timezone"),
        "current": {
            "time": current["time"],
            "temperature_f": current["temperature_2m"],
            "humidity": current["relative_humidity_2m"],
            "wind_mph": current["wind_speed_10m"],
            "wind_direction": current["wind_direction_10m"],
        },
        "rain_past_week_in": round(rain_total, 2),
        "rain_days": len(past_days),
        "dryness": {
            # Vapour pressure deficit (kPa): how hard the air pulls water out of plants
            "vpd_kpa": hourly["vapour_pressure_deficit"][start],
            # Water in the top 1 cm of soil (m³ of water per m³ of soil)
            "soil_moisture": hourly["soil_moisture_0_to_1cm"][start],
        },
        "window": {
            "start": hours[0]["time"],
            "end": hours[-1]["time"],
            "hours": len(hours),
            "min_humidity": min(h["humidity"] for h in hours),
            "max_wind_mph": max(h["wind"] for h in hours),
            "max_gust_mph": max((h["gust"] for h in hours if h["gust"] is not None), default=None),
            "max_temp_f": max(h["temp"] for h in hours),
        },
        "risk": {
            **risk_level(peak["ffwi"]),
            "index": round(peak["ffwi"], 1),
            "peak_time": peak["time"],
            "method": "Fosberg Fire Weather Index, worst hour in the forecast window",
        },
    }


def explain(weather):
    """Plain-language reasons behind the risk level, each tied to a real number."""
    w, d = weather["window"], weather["dryness"]
    rain = weather["rain_past_week_in"]
    reasons = []

    # Each check uses thresholds fire forecasters commonly watch for
    if w["min_humidity"] <= 25:
        reasons.append({"effect": "raises", "text": f"Humidity drops to {w['min_humidity']:.0f}%, so dead grass and brush dry out fast."})
    elif w["min_humidity"] >= 60:
        reasons.append({"effect": "lowers", "text": f"Humid air: humidity stays above {w['min_humidity']:.0f}%."})

    gust = w["max_gust_mph"]
    if w["max_wind_mph"] >= 15 or (gust is not None and gust >= 25):
        gust_text = f" (gusts {gust:.0f} mph)" if gust is not None else ""
        reasons.append({"effect": "raises", "text": f"Winds up to {w['max_wind_mph']:.0f} mph{gust_text} can push flames quickly."})
    elif w["max_wind_mph"] < 8:
        reasons.append({"effect": "lowers", "text": f"Light winds, under {w['max_wind_mph']:.0f} mph."})

    if rain < 0.1:
        reasons.append({"effect": "raises", "text": f"Almost no rain in the past {weather['rain_days']} days ({rain:.2f} in)."})
    elif rain >= 0.5:
        reasons.append({"effect": "lowers", "text": f"{rain:.2f} in of rain fell in the past {weather['rain_days']} days."})

    if d["vpd_kpa"] is not None and d["vpd_kpa"] >= 2.0:
        reasons.append({"effect": "raises", "text": f"High vapour-pressure deficit ({d['vpd_kpa']:.1f} kPa): vegetation is losing moisture quickly."})

    if w["max_temp_f"] >= 85:
        reasons.append({"effect": "raises", "text": f"Hot: up to {w['max_temp_f']:.0f}°F in the next {w['hours']} hours."})
    elif w["max_temp_f"] <= 50:
        reasons.append({"effect": "lowers", "text": f"Cool: no warmer than {w['max_temp_f']:.0f}°F in the next {w['hours']} hours."})

    return reasons


def get_fire_context(lat, lon):
    """Everything the info panel needs for one spot. Parts that fail come back as None."""
    # Round to ~1 km so nearby fires share one cached answer (and one set of API calls)
    key = (round(lat, 2), round(lon, 2))
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_SECONDS:
            return hit[1]

    # Ask both services at the same time instead of one after the other
    with ThreadPoolExecutor(max_workers=2) as pool:
        weather_job = pool.submit(fetch_weather, lat, lon)
        place_job = pool.submit(reverse_geocode, lat, lon)

    result = {"location": None, "weather": None, "reasons": [], "errors": []}
    try:
        result["location"] = place_job.result()
    except Exception as err:  # the panel still works without a place name
        result["errors"].append(f"location: {err}")
    try:
        result["weather"] = summarize_weather(weather_job.result())
        result["reasons"] = explain(result["weather"])
    except Exception as err:
        result["errors"].append(f"weather: {err}")

    # Honest placeholder until SkyAlert's model exists
    result["prediction"] = {
        "probability": None,
        "note": "SkyAlert's prediction model isn't connected yet, so there's no probability to show.",
    }

    # Only cache full answers, so a temporary outage gets retried next time
    if not result["errors"]:
        with _cache_lock:
            _cache[key] = (time.time(), result)
    return result
