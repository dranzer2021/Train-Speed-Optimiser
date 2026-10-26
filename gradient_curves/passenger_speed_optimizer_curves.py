"""Passenger-train whole-route speed-profile energy optimiser (research prototype).

Install: pip install numpy scipy matplotlib
Run:     python passenger_speed_optimizer.py

All sample parameters are illustrative. This is not a certified railway braking,
signalling, or train-protection system. Calibrate and independently validate before use.
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

G = 9.80665
J_PER_KWH = 3.6e6


@dataclass
class Train:
    # Equivalent model for locomotive + all passenger coaches
    mass_kg: float
    davis_A_N: float
    davis_B_N_per_mps: float
    davis_C_N_per_mps2: float
    traction_force_max_N: float
    traction_power_max_W: float
    service_brake_force_max_N: float
    rotating_mass_fraction: float = 0.05
    max_accel_mps2: float = 0.50
    max_decel_mps2: float = 0.65
    traction_efficiency: float = 0.90
    regen_efficiency: float = 0.72
    regen_power_max_W: float = 2_500_000.0
    regen_acceptance: float = 0.60
    auxiliary_power_W: float = 70_000.0
    adhesion_coefficient: float = 0.22
    driven_weight_fraction: float = 0.20
    # Illustrative empirical curve-resistance coefficient; calibrate from railway data.
    # R_curve = coefficient * train_mass / radius. Units: m^2/s^2.
    curve_resistance_coefficient_m2_s2: float = 5.0

    @property
    def effective_mass_kg(self):
        return self.mass_kg * (1 + self.rotating_mass_fraction)

    def resistance_N(self, v):
        v = max(float(v), 0.0)
        return (
            self.davis_A_N
            + self.davis_B_N_per_mps * v
            + self.davis_C_N_per_mps2 * v * v
        )

    def curve_resistance_N(self, radius_m):
        """Approximate additional curve resistance; straight track uses infinite radius.

        This empirical form is a configurable placeholder, not a universal railway law.
        Calibrate the coefficient against measured/coherent route-specific resistance data.
        """
        radius = np.asarray(radius_m, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            resistance = self.curve_resistance_coefficient_m2_s2 * self.mass_kg / radius
        return np.where(np.isfinite(radius), np.maximum(resistance, 0.0), 0.0)

    def traction_limit_N(self, v):
        v = max(float(v), 0.5)
        adhesion = (
            self.adhesion_coefficient * self.driven_weight_fraction * self.mass_kg * G
        )
        return max(
            0.0, min(self.traction_force_max_N, self.traction_power_max_W / v, adhesion)
        )

    def brake_limit_N(self, v):
        adhesion = self.adhesion_coefficient * self.mass_kg * G
        return max(0.0, min(self.service_brake_force_max_N, adhesion))


@dataclass
class Route:
    chainage_m: np.ndarray  # N nodes, increasing
    gradient_permille: np.ndarray  # N-1 intervals; positive = uphill
    speed_limit_mps: np.ndarray  # N nodes
    initial_speed_mps: float = 0.0
    final_speed_mps: float = 0.0
    curve_radius_m: np.ndarray | None = (
        None  # N-1 intervals; np.inf means straight track
    )

    def validate(self):
        self.chainage_m = np.asarray(self.chainage_m, float)
        self.gradient_permille = np.asarray(self.gradient_permille, float)
        self.speed_limit_mps = np.asarray(self.speed_limit_mps, float)
        n = len(self.chainage_m)
        if self.curve_radius_m is None:
            self.curve_radius_m = np.full(n - 1, np.inf, dtype=float)
        else:
            self.curve_radius_m = np.asarray(self.curve_radius_m, float)
        if (
            n < 3
            or len(self.gradient_permille) != n - 1
            or len(self.speed_limit_mps) != n
            or len(self.curve_radius_m) != n - 1
        ):
            raise ValueError(
                "Require N chainages, N-1 gradients and N speed limits; N >= 3."
            )
        if np.any(np.diff(self.chainage_m) <= 0):
            raise ValueError("Chainages must increase strictly.")
        if np.any(np.isnan(self.curve_radius_m)) or np.any(self.curve_radius_m <= 0):
            raise ValueError(
                "Curve radii must be positive; use np.inf for straight track."
            )
        if np.any(self.speed_limit_mps <= 0):
            raise ValueError("Speed limits must be positive.")
        if min(self.initial_speed_mps, self.final_speed_mps) < 0:
            raise ValueError("Endpoint speeds cannot be negative.")
        if (
            self.initial_speed_mps > self.speed_limit_mps[0]
            or self.final_speed_mps > self.speed_limit_mps[-1]
        ):
            raise ValueError("Endpoint speed exceeds speed limit.")


@dataclass
class Result:
    success: bool
    profile: dict
    metrics: dict
    solver_result: object


def evaluate(speed, route, train):
    """Spatial work-energy model; interval force is inferred from adjacent speeds."""
    ds = np.diff(route.chainage_m)
    v0, v1 = speed[:-1], speed[1:]
    vbar = np.maximum((v0 + v1) / 2, 0.1)
    denom = v0 + v1
    dt = np.divide(2 * ds, denom, out=np.full_like(ds, np.inf), where=denom > 1e-7)
    grade = train.mass_kg * G * route.gradient_permille / 1000
    resistance = np.array([train.resistance_N(v) for v in vbar])
    curve_resistance = train.curve_resistance_N(route.curve_radius_m)
    total_resistance = resistance + curve_resistance
    inertial = train.effective_mass_kg * (v1 * v1 - v0 * v0) / (2 * ds)
    f_required = inertial + total_resistance + grade
    f_trac, f_brake = np.maximum(f_required, 0), np.maximum(-f_required, 0)
    e_trac = f_trac * ds / max(train.traction_efficiency, 1e-6)
    e_regen_potential = f_brake * ds * train.regen_efficiency
    e_regen_cap = train.regen_power_max_W * np.where(np.isfinite(dt), dt, 0)
    e_regen = np.minimum(e_regen_potential, e_regen_cap) * np.clip(
        train.regen_acceptance, 0, 1
    )
    e_aux = train.auxiliary_power_W * dt
    accel = (v1 * v1 - v0 * v0) / (2 * ds)
    f_tlim = np.array([train.traction_limit_N(v) for v in vbar])
    f_blim = np.array([train.brake_limit_N(v) for v in vbar])
    return dict(
        ds=ds,
        dt=dt,
        vbar=vbar,
        grade=grade,
        resistance=resistance,
        curve_resistance=curve_resistance,
        total_resistance=total_resistance,
        f_required=f_required,
        f_trac=f_trac,
        f_brake=f_brake,
        f_tlim=f_tlim,
        f_blim=f_blim,
        e_trac=e_trac,
        e_regen=e_regen,
        e_aux=e_aux,
        e_net=e_trac + e_aux - e_regen,
        time_s=float(np.sum(dt)),
        accel=accel,
    )


def optimise(route, train, target_time_s, mode="energy", max_iterations=1500):
    """Direct transcription: node speeds are decision variables; SLSQP handles constraints.

    mode='energy' minimises net kWh. mode='baseline' finds a smooth-reference
    profile under identical constraints, enabling fair same-time benchmarking.
    """
    route.validate()
    if target_time_s <= 0:
        raise ValueError("Target time must be positive.")
    if mode not in ("energy", "baseline"):
        raise ValueError("mode must be 'energy' or 'baseline'")
    n = len(route.chainage_m)
    distance = route.chainage_m[-1] - route.chainage_m[0]
    limits = route.speed_limit_mps
    # A smooth reference is an initial guess and baseline objective, not a safety envelope.
    cruise = min(distance / target_time_s, float(np.min(limits)))
    alpha = (route.chainage_m - route.chainage_m[0]) / distance
    reference = (
        cruise
        + (1 - alpha) * (route.initial_speed_mps - cruise)
        + alpha * (route.final_speed_mps - cruise)
    )
    x0 = np.minimum(np.maximum(reference, 0.1), limits)
    x0[0], x0[-1] = route.initial_speed_mps, route.final_speed_mps
    bounds = [(0.0, float(v)) for v in limits]
    bounds[0] = (route.initial_speed_mps, route.initial_speed_mps)
    bounds[-1] = (route.final_speed_mps, route.final_speed_mps)
    reference = np.maximum(reference, 0.1)

    def objective(v):
        ev = evaluate(v, route, train)
        if not np.isfinite(ev["time_s"]) or not np.all(np.isfinite(ev["e_net"])):
            return 1e12
        if mode == "energy":
            return float(np.sum(ev["e_net"]) / J_PER_KWH)
        return float(np.mean(((v - reference) / np.maximum(limits, 1.0)) ** 2))

    def time_eq(v):
        return evaluate(v, route, train)["time_s"] - target_time_s

    def ineq(v):
        ev = evaluate(v, route, train)
        a = ev["accel"]
        # Scale force slacks for numerical conditioning; all values must be >= 0.
        return np.concatenate(
            (
                (ev["f_tlim"] - ev["f_trac"]) / 1e5,
                (ev["f_blim"] - ev["f_brake"]) / 1e5,
                train.max_accel_mps2 - np.maximum(a, 0),
                train.max_decel_mps2 - np.maximum(-a, 0),
            )
        )

    sol = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": time_eq}, {"type": "ineq", "fun": ineq}],
        options={"maxiter": max_iterations, "ftol": 1e-8, "disp": False},
    )
    v = np.asarray(sol.x)
    ev = evaluate(v, route, train)
    slack = ineq(v)
    time_error = ev["time_s"] - target_time_s
    validated = (
        np.all(np.isfinite(v))
        and np.all(np.isfinite(ev["dt"]))
        and np.min(v) >= -1e-5
        and np.max(v - limits) <= 1e-3
        and abs(time_error) <= max(0.5, 0.002 * target_time_s)
        and np.min(slack) >= -1e-3
    )
    elapsed = np.r_[0.0, np.cumsum(ev["dt"])]
    km = distance / 1000
    profile = dict(
        chainage_m=route.chainage_m.copy(),
        speed_mps=v,
        speed_kmh=v * 3.6,
        speed_limit_kmh=limits * 3.6,
        elapsed_time_s=elapsed,
        gradient_permille=route.gradient_permille.copy(),
        curve_radius_m=route.curve_radius_m.copy(),
        curve_resistance_kN=ev["curve_resistance"] / 1000,
        traction_force_kN=ev["f_trac"] / 1000,
        braking_force_kN=ev["f_brake"] / 1000,
        wheel_traction_power_kW=ev["f_trac"] * ev["vbar"] / 1000,
        accepted_regen_power_kW=ev["e_regen"] / np.maximum(ev["dt"], 1e-9) / 1000,
        interval_net_energy_kWh=ev["e_net"] / J_PER_KWH,
    )
    metrics = dict(
        distance_km=km,
        target_time_s=target_time_s,
        journey_time_s=ev["time_s"],
        time_error_s=time_error,
        traction_energy_kWh=float(np.sum(ev["e_trac"]) / J_PER_KWH),
        curve_resistance_work_kWh=float(
            np.sum(ev["curve_resistance"] * ev["ds"]) / J_PER_KWH
        ),
        accepted_regenerative_energy_kWh=float(np.sum(ev["e_regen"]) / J_PER_KWH),
        auxiliary_energy_kWh=float(np.sum(ev["e_aux"]) / J_PER_KWH),
        net_energy_kWh=float(np.sum(ev["e_net"]) / J_PER_KWH),
        net_energy_kWh_per_km=float(np.sum(ev["e_net"]) / J_PER_KWH / max(km, 1e-9)),
        max_speed_kmh=float(np.max(v) * 3.6),
        max_acceleration_mps2=float(
            np.max(np.maximum(ev["accel"], 0)),
        ),
        max_deceleration_mps2=float(
            np.max(np.maximum(-ev["accel"], 0)),
        ),
        solver_success=bool(sol.success),
        validated=bool(validated),
        solver_message=str(sol.message),
        iterations=int(getattr(sol, "nit", -1)),
        minimum_scaled_constraint_slack=float(np.min(slack)),
    )
    return Result(bool(sol.success and validated), profile, metrics, sol)


def benchmark(route, train, target_time_s):
    """Same-time comparison. Report savings only if both trajectories validate."""
    optimum = optimise(route, train, target_time_s, mode="energy")
    baseline = optimise(route, train, target_time_s, mode="baseline")
    saving = None
    if (
        optimum.success
        and baseline.success
        and abs(baseline.metrics["net_energy_kWh"]) > 1e-9
    ):
        saving = (
            100
            * (baseline.metrics["net_energy_kWh"] - optimum.metrics["net_energy_kWh"])
            / abs(baseline.metrics["net_energy_kWh"])
        )
    return optimum, baseline, saving


def example():
    # Illustrative 30 km route, 500 m mesh. Replace with route survey / timetable data.
    s = np.linspace(0, 30000, 61)
    grade = np.where(
        s[:-1] < 8000,
        0.0,
        np.where(
            s[:-1] < 14000,
            4.0,
            np.where(s[:-1] < 21000, -3.0, np.where(s[:-1] < 26000, 1.0, 0.0)),
        ),
    )
    limits = np.full(len(s), 33.33)  # 120 km/h
    limits[(s >= 10000) & (s <= 13000)] = 25.0  # 90 km/h restriction
    limits[(s >= 22000) & (s <= 24000)] = 19.44  # 70 km/h restriction
    # Example curve geometry: straight by default, with two illustrative curved sections.
    # Replace these radii with surveyed track geometry. Speed limits remain explicit inputs.
    curve_radius = np.full(len(s) - 1, np.inf)
    interval_midpoints = (s[:-1] + s[1:]) / 2
    curve_radius[(interval_midpoints >= 5000) & (interval_midpoints < 8000)] = 800.0
    curve_radius[(interval_midpoints >= 17000) & (interval_midpoints < 19000)] = 350.0
    route = Route(s, grade, limits, 0.0, 0.0, curve_radius)
    train = Train(
        mass_kg=500000.0,
        davis_A_N=8000.0,
        davis_B_N_per_mps=110.0,
        davis_C_N_per_mps2=12.0,
        traction_force_max_N=250000.0,
        traction_power_max_W=6e6,
        service_brake_force_max_N=300000.0,
        max_accel_mps2=0.50,
        max_decel_mps2=0.65,
        traction_efficiency=0.90,
        regen_efficiency=0.72,
        regen_power_max_W=2.5e6,
        regen_acceptance=0.60,
        auxiliary_power_W=70000.0,
        adhesion_coefficient=0.22,
        driven_weight_fraction=0.20,
    )
    return (
        route,
        train,
        1500.0,
    )  # 25 min illustrative target; feasibility must be checked


def run_tests():
    import unittest

    class Tests(unittest.TestCase):
        def setUp(self):
            self.route, self.train, _ = example()

        def test_resistance_increases_with_speed(self):
            self.assertGreater(self.train.resistance_N(25), self.train.resistance_N(10))

        def test_grade_sign(self):
            self.assertGreater(self.train.mass_kg * G * 0.005, 0)
            self.assertLess(self.train.mass_kg * G * (-0.005), 0)

        def test_route_validation(self):
            self.route.validate()
            with self.assertRaises(ValueError):
                Route(
                    np.array([0.0, 10.0, 10.0]),
                    np.array([0.0, 0.0]),
                    np.array([10.0, 10.0, 10.0]),
                ).validate()

        def test_curve_resistance_higher_for_tighter_curve(self):
            r = self.train.curve_resistance_N(np.array([np.inf, 800.0, 400.0]))
            self.assertEqual(float(r[0]), 0.0)
            self.assertGreater(float(r[2]), float(r[1]))

        def test_curve_resistance_increases_required_force(self):
            r = Route(
                np.array([0.0, 100.0, 200.0]),
                np.array([0.0, 0.0]),
                np.array([20.0, 20.0, 20.0]),
                10.0,
                10.0,
                np.array([np.inf, 300.0]),
            )
            straight = evaluate(
                np.array([10.0, 10.0, 10.0]),
                Route(r.chainage_m, r.gradient_permille, r.speed_limit_mps, 10.0, 10.0),
                self.train,
            )
            curved = evaluate(np.array([10.0, 10.0, 10.0]), r, self.train)
            self.assertGreater(curved["f_required"][1], straight["f_required"][1])

        def test_curve_radius_validation(self):
            with self.assertRaises(ValueError):
                Route(
                    np.array([0.0, 10.0, 20.0]),
                    np.array([0.0, 0.0]),
                    np.array([10.0, 10.0, 10.0]),
                    curve_radius_m=np.array([100.0, 0.0]),
                ).validate()

        def test_energy_conversion(self):
            self.assertAlmostEqual(J_PER_KWH / 3.6e6, 1.0)

        def test_regen_acceptance_zero(self):
            r = Route(
                np.array([0.0, 100.0, 200.0]),
                np.array([0.0, 0.0]),
                np.array([20.0, 20.0, 20.0]),
                10.0,
                6.0,
            )
            self.train.regen_acceptance = 0.0
            self.assertEqual(
                float(
                    np.sum(
                        evaluate(np.array([10.0, 8.0, 6.0]), r, self.train)["e_regen"]
                    )
                ),
                0.0,
            )

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    )
    if not result.wasSuccessful():
        raise SystemExit("Tests failed")


def plot(route, optimum, baseline):
    import matplotlib.pyplot as plt

    x = route.chainage_m / 1000
    fig, ax = plt.subplots()
    ax.plot(x, optimum.profile["speed_kmh"], label="Energy optimum")
    ax.plot(x, baseline.profile["speed_kmh"], "--", label="Smooth baseline")
    ax.plot(x, optimum.profile["speed_limit_kmh"], ":", label="Speed limit")
    ax.set(
        xlabel="Chainage (km)",
        ylabel="Speed (km/h)",
        title="Passenger train speed profile",
    )
    ax.grid()
    ax.legend()
    plt.tight_layout()
    fig, ax = plt.subplots()
    ax.plot(
        optimum.profile["elapsed_time_s"][1:] / 60,
        np.cumsum(optimum.profile["interval_net_energy_kWh"]),
        label="Energy optimum",
    )
    ax.plot(
        baseline.profile["elapsed_time_s"][1:] / 60,
        np.cumsum(baseline.profile["interval_net_energy_kWh"]),
        label="Smooth baseline",
    )
    ax.set(
        xlabel="Elapsed time (min)",
        ylabel="Cumulative net energy (kWh)",
        title="Cumulative net electrical energy",
    )
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_tests()
    route, train, target = example()
    print("\nRunning passenger-train whole-route optimisation...")
    optimum, baseline, saving = benchmark(route, train, target)
    for name, result in [("ENERGY OPTIMUM", optimum), ("SMOOTH BASELINE", baseline)]:
        print(f"\n{name}: success={result.success}")
        for k, v in result.metrics.items():
            print(f"{k}: {v}")
    print(
        "\nEnergy saving (%):",
        (
            saving
            if saving is not None
            else "not reported: both profiles must pass validation"
        ),
    )
    if optimum.success and baseline.success:
        try:
            plot(route, optimum, baseline)
        except Exception as exc:
            print("Plotting skipped:", exc)
