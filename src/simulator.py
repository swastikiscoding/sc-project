"""Transparent prototype simulator; not the final research optimizer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.fuzzy_controller import FuzzyPriorityController


def make_synthetic_evs(
    n_evs: int = 25,
    seed: int = 7,
    min_energy_kwh: float = 5.0,
    max_energy_kwh: float = 22.0,
) -> pd.DataFrame:
    """Create synthetic one-day EV sessions.

    Synthetic sessions are not measured data. The departure hour can exceed
    24 to represent next-day departure.
    """
    rng = np.random.default_rng(seed)
    arrival = rng.integers(0, 24, size=n_evs)
    dwell = rng.integers(2, 10, size=n_evs)
    departure = arrival + dwell
    energy = rng.uniform(min_energy_kwh, max_energy_kwh, size=n_evs)
    max_power = rng.choice([7.2, 11.0], size=n_evs, p=[0.7, 0.3])
    return pd.DataFrame({
        "ev_id": [f"EV-{i+1:03d}" for i in range(n_evs)],
        "arrival_hour": arrival,
        "departure_hour": departure,
        "energy_kwh": energy.round(2),
        "max_power_kw": max_power,
    }).sort_values("arrival_hour").reset_index(drop=True)


def normalize_ev_csv(df: pd.DataFrame) -> pd.DataFrame:
    required = {"arrival_hour", "departure_hour", "energy_kwh"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"EV CSV missing required columns: {sorted(missing)}")
    result = df.copy()
    result["arrival_hour"] = pd.to_numeric(result["arrival_hour"], errors="raise")
    result["departure_hour"] = pd.to_numeric(result["departure_hour"], errors="raise")
    result["energy_kwh"] = pd.to_numeric(result["energy_kwh"], errors="raise")
    if "max_power_kw" not in result:
        result["max_power_kw"] = 7.2
    result["max_power_kw"] = pd.to_numeric(result["max_power_kw"], errors="raise")
    if ((result["arrival_hour"] < 0) | (result["arrival_hour"] > 23)).any():
        raise ValueError("arrival_hour must be in [0, 23]")
    if ((result["departure_hour"] < 0) | (result["departure_hour"] > 48)).any():
        raise ValueError("departure_hour must be in [0, 48]")
    if (result["energy_kwh"] <= 0).any() or (result["max_power_kw"] <= 0).any():
        raise ValueError("energy_kwh and max_power_kw must be positive")
    result.loc[result["departure_hour"] <= result["arrival_hour"], "departure_hour"] += 24
    return result


def simulate_day(
    evs: pd.DataFrame,
    pv_kw: np.ndarray,
    grid_capacity_kw: float = 100.0,
    v2g_enabled: bool = False,
    fixed_weights: tuple[float, float, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly heuristic dispatch for pipeline testing.

    When fuzzy mode is active, priorities influence a transparent heuristic.
    This is not an exact solution to the weighted multi-objective problem.
    Replace with a mathematical optimizer for publishable comparisons.
    """
    if len(pv_kw) != 24:
        raise ValueError("pv_kw must contain exactly 24 hourly values")
    controller = FuzzyPriorityController()
    evs = normalize_ev_csv(evs)
    remaining = {row.ev_id: float(row.energy_kwh) for row in evs.itertuples()}
    soc = {row.ev_id: 0.55 for row in evs.itertuples()}
    rows = []
    weight_rows = []

    for hour in range(24):
        active = evs[(evs.arrival_hour <= hour) & (evs.departure_hour > hour)]
        pv = max(0.0, float(pv_kw[hour]))
        remaining_energy = sum(remaining[str(eid)] for eid in active.ev_id)
        renewable_surplus = np.clip((pv - min(remaining_energy, grid_capacity_kw)) / max(grid_capacity_kw, 1), 0, 1)
        grid_stress = np.clip((remaining_energy - pv) / max(grid_capacity_kw, 1), 0, 1)
        # Aggregate urgency: energy deficit divided by remaining connection time
        # relative to a nominal 7.2 kW charger.
        urgency_vals = []
        for row in active.itertuples():
            time_left = max(1, float(row.departure_hour) - hour)
            urgency_vals.append(np.clip(remaining[str(row.ev_id)] / (time_left * float(row.max_power_kw)), 0, 1))
        ev_urgency = float(np.mean(urgency_vals)) if urgency_vals else 0.0

        if fixed_weights is None:
            w = controller.predict(float(grid_stress), float(renewable_surplus), ev_urgency)
            weights = (w["w_grid"], w["w_station"], w["w_user"])
        else:
            weights = np.array(fixed_weights, dtype=float)
            if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
                raise ValueError("fixed_weights must be nonnegative and sum to 1")

        # Heuristic: allocate more of available grid headroom to charging when
        # user priority is high; renewables are used first. This is deliberately
        # a placeholder, not an optimizer.
        available_headroom = max(0.0, grid_capacity_kw - pv)
        user_factor = 0.55 + 0.9 * weights[2]
        grid_budget = min(grid_capacity_kw, available_headroom * user_factor)
        available_power = pv + grid_budget
        ev_power = 0.0

        # Urgent EVs first; departure requirements are still not guaranteed.
        active_sorted = active.copy()
        if not active_sorted.empty:
            active_sorted["urgency_sort"] = active_sorted.apply(
                lambda r: remaining[str(r.ev_id)] /
                max(1.0, (float(r.departure_hour) - hour) * float(r.max_power_kw)), axis=1
            )
            active_sorted = active_sorted.sort_values("urgency_sort", ascending=False)

        for row in active_sorted.itertuples():
            need = remaining[str(row.ev_id)]
            if need <= 1e-6:
                continue
            charge = min(float(row.max_power_kw), need, available_power - ev_power)
            if charge > 0:
                remaining[str(row.ev_id)] -= charge
                ev_power += charge
            elif v2g_enabled and soc[str(row.ev_id)] > 0.35:
                # V2G is illustrative only: no EV exports unless explicitly enabled.
                discharge = min(float(row.max_power_kw), 2.0, available_power * 0.05)
                ev_power -= discharge

        grid_import = max(0.0, ev_power - pv)
        renewable_used = min(pv, max(0.0, ev_power))
        rows.append({
            "hour": hour, "pv_kw_proxy": pv, "ev_charging_kw": max(0.0, ev_power),
            "grid_import_kw": grid_import, "renewable_used_kw": renewable_used,
            "unserved_energy_kwh": sum(remaining[str(eid)] for eid in active.ev_id),
        })
        weight_rows.append({
            "hour": hour, "grid_stress": grid_stress,
            "renewable_surplus": renewable_surplus, "ev_urgency": ev_urgency,
            "w_grid": weights[0], "w_station": weights[1], "w_user": weights[2],
        })

    return pd.DataFrame(rows), pd.DataFrame(weight_rows)
