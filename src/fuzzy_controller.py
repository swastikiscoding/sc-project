"""Mamdani fuzzy inference system for stakeholder priority weights."""
from __future__ import annotations

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyPriorityController:
    """Maps grid stress, renewable surplus, and EV urgency to three weights.

    Inputs are normalized to [0, 1]. Outputs are scores which are normalized
    into weights summing to one. Membership functions/rules are initial
    hypotheses for experimentation, not validated optimal rules.
    """

    def __init__(self):
        self.grid = ctrl.Antecedent(np.linspace(0, 1, 101), "grid_stress")
        self.renewable = ctrl.Antecedent(np.linspace(0, 1, 101), "renewable_surplus")
        self.urgency = ctrl.Antecedent(np.linspace(0, 1, 101), "ev_urgency")

        self.grid_priority = ctrl.Consequent(np.linspace(0, 1, 101), "grid_priority")
        self.station_priority = ctrl.Consequent(np.linspace(0, 1, 101), "station_priority")
        self.user_priority = ctrl.Consequent(np.linspace(0, 1, 101), "user_priority")

        for variable in (self.grid, self.renewable, self.urgency):
            variable["low"] = fuzz.trimf(variable.universe, [0, 0, 0.5])
            variable["medium"] = fuzz.trimf(variable.universe, [0, 0.5, 1])
            variable["high"] = fuzz.trimf(variable.universe, [0.5, 1, 1])

        for variable in (self.grid_priority, self.station_priority, self.user_priority):
            variable["low"] = fuzz.trimf(variable.universe, [0, 0, 0.5])
            variable["medium"] = fuzz.trimf(variable.universe, [0, 0.5, 1])
            variable["high"] = fuzz.trimf(variable.universe, [0.5, 1, 1])

        rules = [
            # Grid stress raises grid priority and lowers station preference.
            ctrl.Rule(self.grid["high"], self.grid_priority["high"]),
            ctrl.Rule(self.grid["medium"], self.grid_priority["medium"]),
            ctrl.Rule(self.grid["low"], self.grid_priority["low"]),

            # Base coverage rules ensure the station output is defined for
            # every renewable input; contextual rules below can strengthen it.
            ctrl.Rule(self.renewable["low"], self.station_priority["low"]),
            ctrl.Rule(self.renewable["medium"], self.station_priority["medium"]),
            ctrl.Rule(self.renewable["high"], self.station_priority["medium"]),
            ctrl.Rule(self.renewable["high"] & self.urgency["low"],
                      self.station_priority["high"]),
            ctrl.Rule(self.renewable["low"] & self.grid["high"],
                      self.station_priority["low"]),
            ctrl.Rule(self.renewable["low"] & self.urgency["high"],
                      self.station_priority["medium"]),

            ctrl.Rule(self.urgency["high"], self.user_priority["high"]),
            ctrl.Rule(self.urgency["medium"], self.user_priority["medium"]),
            ctrl.Rule(self.urgency["low"], self.user_priority["low"]),

            # Conflict cases: preserve grid priority when both grid stress and
            # EV urgency are high; user satisfaction remains represented.
            ctrl.Rule(self.grid["high"] & self.urgency["high"],
                      self.grid_priority["high"]),
            ctrl.Rule(self.grid["low"] & self.urgency["high"],
                      self.user_priority["high"]),
            ctrl.Rule(self.renewable["high"] & self.grid["low"],
                      self.station_priority["high"]),
        ]
        self.system = ctrl.ControlSystem(rules)

    def predict(self, grid_stress: float, renewable_surplus: float, ev_urgency: float) -> dict:
        inputs = {
            "grid_stress": grid_stress,
            "renewable_surplus": renewable_surplus,
            "ev_urgency": ev_urgency,
        }
        for name, value in inputs.items():
            if not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be between 0 and 1; got {value}")

        sim = ctrl.ControlSystemSimulation(self.system, clip_to_bounds=True)
        for name, value in inputs.items():
            sim.input[name] = float(value)
        sim.compute()

        scores = np.array([
            sim.output["grid_priority"],
            sim.output["station_priority"],
            sim.output["user_priority"],
        ], dtype=float)
        scores = np.maximum(scores, 1e-6)
        weights = scores / scores.sum()
        return {
            "grid_score": float(scores[0]),
            "station_score": float(scores[1]),
            "user_score": float(scores[2]),
            "w_grid": float(weights[0]),
            "w_station": float(weights[1]),
            "w_user": float(weights[2]),
        }
