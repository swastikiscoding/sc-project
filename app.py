from __future__ import annotations

import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.nasa_power import fetch_hourly_weather, add_synthetic_pv_profile
from src.simulator import make_synthetic_evs, normalize_ev_csv, simulate_day

st.set_page_config(page_title="Fuzzy-Adaptive EV Charging & V2G", layout="wide")
st.title("Fuzzy-Adaptive EV Charging & V2G")
st.caption("Research prototype v0.1 — fuzzy priority controller + hourly simulation")

with st.sidebar:
    st.header("Scenario settings")
    latitude = st.number_input("Latitude", value=28.7041, min_value=-90.0, max_value=90.0, format="%.4f")
    longitude = st.number_input("Longitude", value=77.1025, min_value=-180.0, max_value=180.0, format="%.4f")
    start_date = st.date_input("Weather start date", value=date(2024, 5, 1))
    n_evs = st.slider("Number of EVs", 5, 80, 25, 5)
    pv_capacity = st.slider("PV capacity (kW)", 10, 150, 50, 10)
    grid_capacity = st.slider("Grid import capacity (kW)", 20, 200, 100, 10)
    seed = st.number_input("Synthetic scenario seed", min_value=0, max_value=100000, value=7)
    use_v2g = st.checkbox("Enable illustrative V2G dispatch", value=False)
    use_live_api = st.checkbox("Fetch hourly data from NASA POWER", value=True)
    uploaded = st.file_uploader("Optional normalized EV sessions CSV", type=["csv"])

if "weather_cache" not in st.session_state:
    st.session_state.weather_cache = None
if "weather_key" not in st.session_state:
    st.session_state.weather_key = None

weather_key = (latitude, longitude, str(start_date))
if use_live_api:
    if st.button("Fetch NASA POWER data") or st.session_state.weather_key != weather_key:
        end_date = start_date + timedelta(days=1)
        try:
            with st.spinner("Requesting NASA POWER hourly observations..."):
                weather = fetch_hourly_weather(
                    latitude, longitude, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
                )
                weather = add_synthetic_pv_profile(weather, float(pv_capacity))
                st.session_state.weather_cache = weather
                st.session_state.weather_key = weather_key
        except Exception as exc:
            st.error(f"NASA POWER request failed: {exc}")
else:
    st.session_state.weather_cache = None

weather = st.session_state.weather_cache
if weather is not None and "pv_power_kw_proxy" in weather.columns:
    pv_profile = weather["pv_power_kw_proxy"].groupby(weather.index.hour).mean().reindex(range(24), fill_value=0).to_numpy()
    if len(pv_profile) != 24:
        pv_profile = np.resize(pv_profile, 24)
    weather_display = weather.reset_index()
else:
    # Explicit synthetic fallback; not presented as observed weather.
    hours = np.arange(24)
    pv_profile = float(pv_capacity) * np.maximum(0, np.sin(np.pi * (hours - 6) / 12)) * 0.75
    weather_display = None
    if use_live_api:
        st.info("No API data loaded yet. Click 'Fetch NASA POWER data'; synthetic PV is used until then.")

if uploaded is not None:
    try:
        evs = normalize_ev_csv(pd.read_csv(uploaded))
        st.sidebar.success(f"Loaded {len(evs)} EV sessions")
    except Exception as exc:
        st.sidebar.error(f"Could not load EV CSV: {exc}")
        evs = make_synthetic_evs(int(n_evs), int(seed))
else:
    evs = make_synthetic_evs(int(n_evs), int(seed))

adaptive_results, adaptive_weights = simulate_day(
    evs, pv_profile, float(grid_capacity), v2g_enabled=use_v2g
)
fixed_results, fixed_weights = simulate_day(
    evs, pv_profile, float(grid_capacity), v2g_enabled=False,
    fixed_weights=(0.34, 0.33, 0.33)
)

unserved = float(adaptive_results["unserved_energy_kwh"].iloc[-1])
renewable = float(adaptive_results["renewable_used_kw"].sum())
grid_energy = float(adaptive_results["grid_import_kw"].sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("EV sessions", len(evs))
c2.metric("PV proxy energy (kWh)", f"{pv_profile.sum():.1f}")
c3.metric("Grid import proxy (kWh)", f"{grid_energy:.1f}")
c4.metric("Remaining active-session demand proxy (kWh)", f"{unserved:.1f}")

st.subheader("1. Renewable profile")
pv_df = pd.DataFrame({"Hour": np.arange(24), "PV power proxy (kW)": pv_profile})
st.plotly_chart(px.area(pv_df, x="Hour", y="PV power proxy (kW)", title="Hourly PV profile (observed-weather-derived proxy or synthetic fallback)"), use_container_width=True)
if weather_display is not None:
    st.caption("Weather data came from NASA POWER. The PV conversion is a simplified proxy and is not a validated PV yield model.")

st.subheader("2. Adaptive controller outputs")
w_long = adaptive_weights.melt(
    id_vars=["hour"], value_vars=["w_grid", "w_station", "w_user"],
    var_name="Stakeholder", value_name="Weight"
)
st.plotly_chart(px.line(w_long, x="hour", y="Weight", color="Stakeholder", markers=True, title="Fuzzy-adaptive stakeholder weights"), use_container_width=True)
st.plotly_chart(px.line(adaptive_weights, x="hour", y=["grid_stress", "renewable_surplus", "ev_urgency"], title="Normalized fuzzy-controller inputs"), use_container_width=True)

st.subheader("3. Dispatch proxy: adaptive vs fixed weights")
comparison = adaptive_results[["hour", "ev_charging_kw", "grid_import_kw"]].merge(
    fixed_results[["hour", "ev_charging_kw", "grid_import_kw"]],
    on="hour", suffixes=("_adaptive", "_fixed")
)
fig = go.Figure()
fig.add_trace(go.Scatter(x=comparison.hour, y=comparison.ev_charging_kw_adaptive, name="Charging — fuzzy"))
fig.add_trace(go.Scatter(x=comparison.hour, y=comparison.ev_charging_kw_fixed, name="Charging — fixed"))
fig.add_trace(go.Scatter(x=comparison.hour, y=comparison.grid_import_kw_adaptive, name="Grid import — fuzzy", line={"dash": "dash"}))
fig.add_trace(go.Scatter(x=comparison.hour, y=comparison.grid_import_kw_fixed, name="Grid import — fixed", line={"dash": "dash"}))
fig.update_layout(title="Heuristic dispatch comparison (not an optimizer result)", xaxis_title="Hour", yaxis_title="Power (kW)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("4. EV session table")
st.dataframe(evs, use_container_width=True, hide_index=True)

st.warning(
    "Research caution: the dispatch policy is a heuristic pipeline test, not COA or a constrained optimizer. "
    "The current V2G option is illustrative and does not yet implement a complete bidirectional SoC/energy accounting model. "
    "Do not use these outputs as evidence of research improvement. The next step is to replace the heuristic with "
    "the exact fixed-weight and fuzzy-adaptive optimization formulations and validate constraints."
)

csv = adaptive_results.to_csv(index=False).encode("utf-8")
st.download_button("Download prototype results CSV", csv, "prototype_dispatch_results.csv", "text/csv")
