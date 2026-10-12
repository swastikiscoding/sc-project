# Fuzzy-Adaptive EV Charging & V2G — v0.1

A research prototype connecting:
- NASA POWER hourly weather/solar API
- Synthetic EV arrival/departure scenarios (or CSV import)
- Mamdani fuzzy controller for adaptive stakeholder weights
- Simple EV charging/V2G dispatch simulator
- Streamlit + Plotly dashboard

## Important research limitations
This is a prototype, not yet a reproduction of the base paper's Cheetah Optimization Algorithm (COA), battery degradation model, or IEEE 33-bus power-flow evaluation. The current dispatch policy is a transparent heuristic used to validate the data/controller/UI pipeline. Do not report its results as paper results.

The NASA POWER solar-radiation field is an irradiance proxy, not PV electrical power. The app converts it to a configurable synthetic PV power profile using a simple scaling assumption. Replace this with a documented PV model before final experiments.

## Setup (Linux/WSL)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## NASA POWER source
Official hourly API documentation:
https://power.larc.nasa.gov/docs/services/api/temporal/hourly/

The app requests `ALLSKY_SFC_SW_DWN` (all-sky surface shortwave downward irradiance) and `T2M` (2-m air temperature) from the hourly point endpoint. API data availability and parameter definitions should be checked against NASA POWER's current documentation.

## ACN-Data
Dataset and API information: https://ev.caltech.edu/dataset

To use session data, export/download a CSV and load it from the sidebar. Column names vary by export. This prototype accepts a normalized CSV with:
- `arrival_hour` (0–23)
- `departure_hour` (0–24; values <= arrival are treated as next-day departures)
- `energy_kwh` (energy required during the session)

If you use raw ACN-Data exports, map the dataset's documented columns to these normalized columns before importing. Do not assume a raw export already follows this schema.

## Next research milestones
1. Reproduce the base paper's objective functions, constraints and COA configuration.
2. Replace heuristic dispatch with a tested optimization formulation.
3. Add pandapower IEEE 33-bus power-flow evaluation.
4. Add uncertainty scenarios and fair fixed-weight/fuzzy-adaptive/NSGA-II comparisons.
5. Validate degradation costs and report all assumptions.
# sc-project
# sc-project
