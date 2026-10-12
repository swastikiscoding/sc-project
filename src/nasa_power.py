"""NASA POWER hourly API client.

Official docs:
https://power.larc.nasa.gov/docs/services/api/temporal/hourly/
"""
from __future__ import annotations

import requests
import pandas as pd

BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"


def fetch_hourly_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timeout: int = 45,
) -> pd.DataFrame:
    """Fetch hourly NASA POWER data.

    Dates must be YYYYMMDD. The API commonly returns JSON metadata and
    parameter values keyed by timestamp. This function parses that structure
    and returns a UTC-indexed dataframe where possible.
    """
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,T2M",
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
        "time-standard": "UTC",
    }
    response = requests.get(BASE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    try:
        parameter_data = payload["properties"]["parameter"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Unexpected NASA POWER response structure: {json_keys(payload)}"
        ) from exc

    frame = pd.DataFrame(parameter_data)
    # POWER timestamps are YYYYMMDDHH strings.
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d%H", errors="coerce")
    frame.index.name = "time"
    frame = frame.sort_index()
    frame = frame.rename(columns={
        "ALLSKY_SFC_SW_DWN": "solar_irradiance",
        "T2M": "temperature_c",
    })
    # POWER may use -999 for unavailable values.
    frame = frame.replace(-999, float("nan")).apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(how="all")
    if frame.empty:
        raise ValueError("NASA POWER returned no usable hourly observations.")
    return frame


def json_keys(payload: dict) -> str:
    return ", ".join(map(str, payload.keys())) if isinstance(payload, dict) else type(payload).__name__


def add_synthetic_pv_profile(
    weather: pd.DataFrame,
    pv_capacity_kw: float = 50.0,
) -> pd.DataFrame:
    """Convert solar irradiance proxy into a transparent, simplified PV profile.

    This is only a prototype approximation. It scales observed irradiance by
    PV capacity and clips to [0, capacity]. It is NOT a validated PV performance
    model and does not model tilt, temperature derating, inverter efficiency,
    shading, or module characteristics.
    """
    result = weather.copy()
    irradiance = result["solar_irradiance"].clip(lower=0).fillna(0)
    # NASA's ALLSKY_SFC_SW_DWN is typically an hourly energy/irradiance quantity
    # in the selected API community's units; this normalization is a deliberately
    # simple proxy and must be replaced by a unit-verified PV model for research.
    normalized = (irradiance / 1000.0).clip(0, 1)
    result["pv_power_kw_proxy"] = normalized * pv_capacity_kw
    return result
