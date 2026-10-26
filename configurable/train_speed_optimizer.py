"""
Configurable passenger-train speed-profile energy optimiser.

Install:
    pip install numpy scipy matplotlib

Run:
    python train_speed_optimizer.py --config config.json

All route restrictions, gradients and curves are supplied in config.json.
This is a research prototype, not a certified railway safety or signalling system.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

J_PER_KWH = 3.6e6


@dataclass
class Train:
    mass_kg: float
    rotating_mass_factor: float
    max_power_w: float
    max_tractive_effort_n: float
    max_brake_force_n: float
    max_acceleration: float
    max_service_decel: float
    max_line_speed_mps: float
    davis_a_n: float
    davis_b_n_per_mps: float
    davis_c_n_per_mps2: float
    traction_efficiency: float
    regen_efficiency: float
    aux_power_w: float
    curve_lateral_accel: float
    gravity: float
    adhesion_coefficient: float
    driven_weight_fraction: float
    regen_power_max_w: float
    regen_acceptance: float

    @property
    def effective_mass_kg(self) -> float:
        # The configured factor is a multiplier (e.g. 1.08), not an added fraction.
        return self.mass_kg * self.rotating_mass_factor

    def resistance_n(self, speed_mps: float) -> float:
        v = max(float(speed_mps), 0.0)
        return self.davis_a_n + self.davis_b_n_per_mps * v + self.davis_c_n_per_mps2 * v * v

    def traction_limit_n(self, speed_mps: float) -> float:
        v = max(float(speed_mps), 0.5)
        adhesion = (
            self.adhesion_coefficient * self.driven_weight_fraction
            * self.mass_kg * self.gravity
        )
        return max(0.0, min(self.max_tractive_effort_n, self.max_power_w / v, adhesion))

    def brake_limit_n(self) -> float:
        adhesion = self.adhesion_coefficient * self.mass_kg * self.gravity
        return max(0.0, min(self.max_brake_force_n, adhesion))


def parse_clock(value: str) -> int | None:
    """Parse HH:MM to minutes after midnight; SOURCE/DESTINATION return None."""
    if value.upper() in {"SOURCE", "DESTINATION", ""}:
        return None
    h, m = value.split(":")
    h, m = int(h), int(m)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Invalid timetable clock value: {value}")
    return h * 60 + m


def load_config(path: str | Path) -> tuple[dict[str, Any], Train]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    t = cfg["train"]

    required = [
        "total_mass_tonne", "rotating_mass_factor", "max_power_kw",
        "max_tractive_effort_kn", "max_brake_force_kn", "max_acceleration_mps2",
        "max_service_decel_mps2", "max_line_speed_kmh", "davis_a_n",
        "davis_b_n_per_mps", "davis_c_n_per_mps2", "traction_efficiency",
        "regen_efficiency", "aux_power_kw", "curve_lateral_accel_mps2",
        "gravity_mps2"
    ]
    missing = [k for k in required if k not in t]
    if missing:
        raise ValueError(f"Missing train configuration keys: {missing}")

    for key in ("traction_efficiency", "regen_efficiency", "regen_acceptance"):
        if key in t and not 0 <= float(t[key]) <= 1:
            raise ValueError(f"{key} must be between 0 and 1")

    train = Train(
        mass_kg=float(t["total_mass_tonne"]) * 1000,
        rotating_mass_factor=float(t["rotating_mass_factor"]),
        max_power_w=float(t["max_power_kw"]) * 1000,
        max_tractive_effort_n=float(t["max_tractive_effort_kn"]) * 1000,
        max_brake_force_n=float(t["max_brake_force_kn"]) * 1000,
        max_acceleration=float(t["max_acceleration_mps2"]),
        max_service_decel=float(t["max_service_decel_mps2"]),
        max_line_speed_mps=float(t["max_line_speed_kmh"]) / 3.6,
        davis_a_n=float(t["davis_a_n"]),
        davis_b_n_per_mps=float(t["davis_b_n_per_mps"]),
        davis_c_n_per_mps2=float(t["davis_c_n_per_mps2"]),
        traction_efficiency=float(t["traction_efficiency"]),
        regen_efficiency=float(t["regen_efficiency"]),
        aux_power_w=float(t["aux_power_kw"]) * 1000,
        curve_lateral_accel=float(t["curve_lateral_accel_mps2"]),
        gravity=float(t["gravity_mps2"]),
        adhesion_coefficient=float(t.get("adhesion_coefficient", 0.22)),
        driven_weight_fraction=float(t.get("driven_weight_fraction", 0.20)),
        regen_power_max_w=float(t.get("regen_power_max_kw", 2500)) * 1000,
        regen_acceptance=float(t.get("regen_acceptance", 0.60)),
    )
    if train.mass_kg <= 0 or train.rotating_mass_factor < 1:
        raise ValueError("Mass must be positive and rotating_mass_factor must be >= 1.")
    return cfg, train


def segment_value(segments: list[dict], midpoint_km: float, key: str, default: float) -> float:
    """Return the value of the segment covering midpoint; otherwise default."""
    matches = [
        seg for seg in segments
        if float(seg["from_km"]) <= midpoint_km < float(seg["to_km"])
    ]
    if len(matches) > 1:
        # Overlap can be intentional only if the input explicitly allows it; safer to reject.
        raise ValueError(f"Overlapping {key} segments at chainage {midpoint_km:.3f} km")
    return float(matches[0][key]) if matches else default


def build_route(cfg: dict[str, Any], train: Train) -> dict[str, np.ndarray | list]:
    route_cfg = cfg["route"]
    stations = route_cfg["stations"]
    if len(stations) < 2:
        raise ValueError("At least source and destination stations are required.")

    station_km = np.array([float(s["km"]) for s in stations])
    if np.any(np.diff(station_km) <= 0):
        raise ValueError("Station distances must increase strictly.")
    start_km, end_km = float(station_km[0]), float(station_km[-1])
    if abs(start_km) > 1e-9:
        raise ValueError("This configuration expects the source station at km 0.")

    opt = cfg.get("optimizer", {})
    spacing = float(opt.get("mesh_spacing_km", 2.0))
    if spacing <= 0:
        raise ValueError("mesh_spacing_km must be positive.")

    # Include station positions and every segment boundary as mesh nodes.
    boundaries = [start_km, end_km]
    for group in ("speed_restrictions", "gradients", "curves"):
        for seg in route_cfg.get(group, []):
            a, b = float(seg["from_km"]), float(seg["to_km"])
            if not (start_km <= a < b <= end_km):
                raise ValueError(f"Invalid {group} range [{a}, {b}] km")
            boundaries.extend([a, b])

    regular = np.arange(start_km, end_km, spacing)
    chainage_km = np.unique(np.round(np.r_[regular, station_km, boundaries, end_km], 8))
    chainage_m = chainage_km * 1000
    midpoint_km = (chainage_km[:-1] + chainage_km[1:]) / 2

    restrictions = route_cfg.get("speed_restrictions", [])
    gradients = route_cfg.get("gradients", [])
    curves = route_cfg.get("curves", [])

    interval_speed_limit = np.full(len(midpoint_km), train.max_line_speed_mps)
    for i, mid in enumerate(midpoint_km):
        limit_kmh = segment_value(restrictions, mid, "speed_kmh", train.max_line_speed_mps * 3.6)
        interval_speed_limit[i] = min(train.max_line_speed_mps, limit_kmh / 3.6)

    gradient_permille = np.array([
        segment_value(gradients, mid, "gradient_permille", 0.0)
        for mid in midpoint_km
    ])

    curve_degree = np.array([
        segment_value(curves, mid, "degree", 0.0)
        for mid in midpoint_km
    ])
    radius_m = np.full(len(midpoint_km), np.inf)
    model = opt.get("curve_radius_model", "indian_railway_approx")
    constant = float(opt.get("curve_radius_constant_m_degree", 1750.0))
    for i, degree in enumerate(curve_degree):
        if degree < 0:
            raise ValueError("Curve degree must be non-negative; use a separate direction field if needed.")
        if degree > 0:
            if model != "indian_railway_approx":
                raise ValueError(f"Unsupported curve_radius_model: {model}")
            radius_m[i] = constant / degree

    curve_limit_mps = np.full(len(midpoint_km), np.inf)
    curved = np.isfinite(radius_m)
    curve_limit_mps[curved] = np.sqrt(train.curve_lateral_accel * radius_m[curved])
    interval_speed_limit = np.minimum(interval_speed_limit, curve_limit_mps)

    # A node must obey the lower limit of adjacent intervals, so a restriction
    # is not bypassed merely because it starts/ends between original grid points.
    node_limit = np.full(len(chainage_km), train.max_line_speed_mps)
    node_limit[:-1] = np.minimum(node_limit[:-1], interval_speed_limit)
    node_limit[1:] = np.minimum(node_limit[1:], interval_speed_limit)

    station_indices = []
    for km in station_km:
        idx = int(np.argmin(np.abs(chainage_km - km)))
        if abs(chainage_km[idx] - km) > 1e-6:
            raise ValueError(f"Station at {km} km is not present in mesh.")
        station_indices.append(idx)

    target_leg_times_s = []
    for i in range(1, len(stations)):
        prev_dep = parse_clock(stations[i - 1]["departure"])
        next_arr = parse_clock(stations[i]["arrival"])
        if prev_dep is None or next_arr is None:
            raise ValueError("Every non-destination departure and every non-source arrival must be HH:MM.")
        # Resolve overnight times by moving the arrival to the next day when needed.
        while next_arr <= prev_dep:
            next_arr += 24 * 60
        target_leg_times_s.append((next_arr - prev_dep) * 60.0)

    return {
        "chainage_km": chainage_km,
        "chainage_m": chainage_m,
        "ds_m": np.diff(chainage_m),
        "midpoint_km": midpoint_km,
        "gradient_permille": gradient_permille,
        "curve_degree": curve_degree,
        "curve_radius_m": radius_m,
        "interval_speed_limit_mps": interval_speed_limit,
        "node_speed_limit_mps": node_limit,
        "station_indices": np.array(station_indices, dtype=int),
        "station_km": station_km,
        "stations": stations,
        "target_leg_times_s": np.array(target_leg_times_s),
    }


def evaluate(speed: np.ndarray, route: dict, train: Train) -> dict[str, np.ndarray | float]:
    ds = np.asarray(route["ds_m"])
    v0, v1 = speed[:-1], speed[1:]
    denom = v0 + v1
    dt = np.divide(
        2 * ds, denom, out=np.full_like(ds, np.inf, dtype=float), where=denom > 1e-7
    )
    vbar = np.maximum((v0 + v1) / 2, 0.1)
    grade = train.mass_kg * train.gravity * np.asarray(route["gradient_permille"]) / 1000
    resistance = np.array([train.resistance_n(v) for v in vbar])
    accel = (v1 * v1 - v0 * v0) / (2 * ds)
    inertial = train.effective_mass_kg * accel
    f_required = inertial + resistance + grade
    f_trac = np.maximum(f_required, 0)
    f_brake = np.maximum(-f_required, 0)

    traction_energy = f_trac * ds / max(train.traction_efficiency, 1e-6)
    regen_potential = f_brake * ds * train.regen_efficiency
    regen_cap = train.regen_power_max_w * np.where(np.isfinite(dt), dt, 0)
    regen = np.minimum(regen_potential, regen_cap) * np.clip(train.regen_acceptance, 0, 1)
    aux = train.aux_power_w * dt
    net = traction_energy + aux - regen
    f_tlim = np.array([train.traction_limit_n(v) for v in vbar])
    f_blim = np.full_like(f_tlim, train.brake_limit_n())

    station_idx = np.asarray(route["station_indices"])
    leg_times = []
    for a, b in zip(station_idx[:-1], station_idx[1:]):
        leg_times.append(float(np.sum(dt[a:b])))

    return {
        "dt": dt, "vbar": vbar, "grade_n": grade, "resistance_n": resistance,
        "accel": accel, "f_required_n": f_required, "f_trac_n": f_trac,
        "f_brake_n": f_brake, "f_tlim_n": f_tlim, "f_blim_n": f_blim,
        "traction_energy_j": traction_energy, "regen_energy_j": regen,
        "aux_energy_j": aux, "net_energy_j": net,
        "time_s": float(np.sum(dt)), "leg_times_s": np.asarray(leg_times),
    }


def optimise(cfg: dict, train: Train, route: dict) -> dict:
    opt = cfg.get("optimizer", {})
    mode = opt.get("mode", "energy")
    if mode not in {"energy", "baseline"}:
        raise ValueError("optimizer.mode must be 'energy' or 'baseline'")

    x_m = np.asarray(route["chainage_m"])
    limits = np.asarray(route["node_speed_limit_mps"])
    station_idx = np.asarray(route["station_indices"])
    target_leg_times = np.asarray(route["target_leg_times_s"])

    # Initial guess: use scheduled average speed per leg, capped by line/curve limits.
    x0 = np.zeros_like(x_m)
    for leg, (a, b) in enumerate(zip(station_idx[:-1], station_idx[1:])):
        distance = x_m[b] - x_m[a]
        cruise = distance / target_leg_times[leg]
        x0[a:b + 1] = np.minimum(cruise, limits[a:b + 1])
    x0[station_idx] = 0.0
    # Avoid adjacent zero speeds except at route endpoints/stations; use a gentle interior seed.
    for idx in range(len(x0)):
        if idx not in set(station_idx.tolist()) and x0[idx] < 0.1:
            x0[idx] = min(0.1, limits[idx])

    bounds = [(0.0, float(lim)) for lim in limits]
    for idx in station_idx:
        bounds[int(idx)] = (0.0, 0.0)

    def objective(v):
        ev = evaluate(v, route, train)
        if not np.all(np.isfinite(ev["dt"])) or not np.all(np.isfinite(ev["net_energy_j"])):
            return 1e15
        if mode == "energy":
            return float(np.sum(ev["net_energy_j"]) / J_PER_KWH)
        # Reference profile for baseline: piecewise constant cruise, with station stops.
        ref = x0
        return float(np.mean(((v - ref) / np.maximum(limits, 1.0)) ** 2))

    def time_eq(v):
        ev = evaluate(v, route, train)
        return (ev["leg_times_s"] - target_leg_times) / 60.0

    def ineq(v):
        ev = evaluate(v, route, train)
        a = np.asarray(ev["accel"])
        return np.concatenate([
            (np.asarray(ev["f_tlim_n"]) - np.asarray(ev["f_trac_n"])) / 1e5,
            (np.asarray(ev["f_blim_n"]) - np.asarray(ev["f_brake_n"])) / 1e5,
            train.max_acceleration - np.maximum(a, 0),
            train.max_service_decel - np.maximum(-a, 0),
        ])

    sol = minimize(
        objective, x0, method="SLSQP", bounds=bounds,
        constraints=[
            {"type": "eq", "fun": time_eq},
            {"type": "ineq", "fun": ineq},
        ],
        options={
            "maxiter": int(opt.get("max_iterations", 1500)),
            "ftol": float(opt.get("ftol", 1e-7)),
            "disp": True,
        },
    )

    v = np.asarray(sol.x)
    ev = evaluate(v, route, train)
    slack = ineq(v)
    time_errors = np.asarray(ev["leg_times_s"]) - target_leg_times
    validated = (
        np.all(np.isfinite(v))
        and np.all(np.isfinite(ev["dt"]))
        and np.min(v) >= -1e-5
        and np.max(v - limits) <= 1e-3
        and np.max(np.abs(time_errors)) <= 2.0
        and np.min(slack) >= -1e-3
        and np.all(np.abs(v[station_idx]) <= 1e-6)
    )

    return {
        "success": bool(sol.success and validated),
        "solver_success": bool(sol.success),
        "validated": bool(validated),
        "message": str(sol.message),
        "iterations": int(getattr(sol, "nit", -1)),
        "chainage_km": x_m / 1000,
        "speed_kmh": v * 3.6,
        "speed_limit_kmh": limits * 3.6,
        "station_indices": station_idx,
        "metrics": {
            "distance_km": float((x_m[-1] - x_m[0]) / 1000),
            "journey_time_min": float(ev["time_s"] / 60),
            "target_running_time_min": float(np.sum(target_leg_times) / 60),
            "traction_energy_kwh": float(np.sum(ev["traction_energy_j"]) / J_PER_KWH),
            "regenerative_energy_kwh": float(np.sum(ev["regen_energy_j"]) / J_PER_KWH),
            "auxiliary_energy_kwh": float(np.sum(ev["aux_energy_j"]) / J_PER_KWH),
            "net_energy_kwh": float(np.sum(ev["net_energy_j"]) / J_PER_KWH),
            "max_speed_kmh": float(np.max(v) * 3.6),
            "max_acceleration_mps2": float(np.max(np.maximum(ev["accel"], 0))),
            "max_deceleration_mps2": float(np.max(np.maximum(-ev["accel"], 0))),
            "max_abs_leg_time_error_s": float(np.max(np.abs(time_errors))),
            "minimum_scaled_constraint_slack": float(np.min(slack)),
        },
        "leg_results": [
            {
                "from_station": route["stations"][i]["code"],
                "to_station": route["stations"][i + 1]["code"],
                "target_running_min": float(target_leg_times[i] / 60),
                "model_running_min": float(ev["leg_times_s"][i] / 60),
                "time_error_s": float(time_errors[i]),
            }
            for i in range(len(target_leg_times))
        ],
        "profile": {
            "chainage_km": (x_m / 1000).tolist(),
            "speed_kmh": (v * 3.6).tolist(),
            "speed_limit_kmh": (limits * 3.6).tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", help="Path to JSON configuration")
    parser.add_argument("--output", default="result.json", help="Path to JSON result")
    parser.add_argument("--plot", action="store_true", help="Plot speed profile (requires matplotlib)")
    args = parser.parse_args()

    cfg, train = load_config(args.config)
    route = build_route(cfg, train)
    result = optimise(cfg, train, route)

    print(f"Optimisation success: {result['success']}")
    print(f"Solver success: {result['solver_success']}")
    print(f"Validated: {result['validated']}")
    print(f"Solver message: {result['message']}")
    print(json.dumps(result["metrics"], indent=2))
    print("\nStation-to-station running times:")
    for leg in result["leg_results"]:
        print(
            f"{leg['from_station']} -> {leg['to_station']}: "
            f"target={leg['target_running_min']:.2f} min, "
            f"model={leg['model_running_min']:.2f} min, "
            f"error={leg['time_error_s']:.1f} s"
        )

    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nDetailed result written to: {args.output}")

    if args.plot:
        import matplotlib.pyplot as plt
        x = result["chainage_km"]
        plt.figure()
        plt.plot(x, result["speed_kmh"], label="Optimised speed")
        plt.plot(x, result["speed_limit_kmh"], ":", label="Node speed limit")
        for station in route["stations"]:
            plt.axvline(float(station["km"]), alpha=0.25)
        plt.xlabel("Chainage (km)")
        plt.ylabel("Speed (km/h)")
        plt.title("Train speed profile")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
