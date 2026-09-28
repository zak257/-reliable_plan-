#!/usr/bin/env python3
"""Run the direct Gurobi resilience MILP; retain the old search for history."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import time
import os
import sys
import hashlib
from dataclasses import asdict
from pathlib import Path

from polar_reliability_planning.resilience_v2.model import (
    Capacity,
    CAPACITY_STEPS,
    assert_discrete_capacity,
    ResilienceConfig,
    evaluate_capacity,
    generate_synthetic_year,
    make_scenarios,
    plan_capacity,
    required_ups_bridge_kwh,
    required_ups_bridge_kw,
    simulate,
    validate_capacity,
    write_plan_outputs,
)


def global_capacity_grid(year, cfg: ResilienceConfig, investment_budget_yuan: float) -> list[Capacity]:
    """Materialized oracle for tiny tests only; production uses the iterator."""
    _, iterator = global_capacity_grid_iterator(year, cfg, investment_budget_yuan)
    return list(iterator)


def global_capacity_grid_iterator(year, cfg: ResilienceConfig, investment_budget_yuan: float):
    """Return the complete fine-grid iterator without materialising it."""
    levels = capacity_grid_levels(year, cfg, investment_budget_yuan)
    return levels, iter_capacity_grid(
        levels, cfg, investment_budget_yuan,
        required_ups_bridge_kwh(year, cfg), required_ups_bridge_kw(year, cfg),
    )


def _discrete_levels(maximum: float, step: float) -> tuple[float, ...]:
    """Return every discrete module level from zero through the data range."""
    count = int(math.ceil(maximum / step - 1e-12))
    return tuple(float(step * i) for i in range(count + 1))


def capacity_grid_levels(year, cfg: ResilienceConfig, investment_budget_yuan: float) -> dict[str, tuple[float, ...] | tuple[int, ...]]:
    """Build every module level affordable under a valid incumbent bound.

    The bound is an economic search bound, not a physical upper limit. Since
    all operating and loss costs are nonnegative, a design whose investment
    alone is above a feasible incumbent objective cannot improve it. Thus the
    finite domain contains every possible improving discrete design while
    avoiding arbitrary peak-based limits.
    """
    if not math.isfinite(investment_budget_yuan) or investment_budget_yuan <= 0:
        raise ValueError("investment_budget_yuan must be a positive finite bound")
    unit_costs = {
        "wind_kw": 1500.0 * 100.0,
        "pv_kw": 1000.0 * 100.0,
        "diesel_units": 180000.0,
        "battery_kwh": 250.0 * 50.0,
        "pcs_kw": 400.0 * 50.0,
        "ups_kwh": 450.0 * 50.0,
        "ups_kw": 250.0 * 50.0,
    }
    def levels_for(name: str, step: float) -> tuple[float, ...]:
        module_count = int(math.floor(investment_budget_yuan / unit_costs[name] + 1e-9))
        return tuple(float(step * i) for i in range(module_count + 1))

    diesel_max = int(math.floor(investment_budget_yuan / unit_costs["diesel_units"] + 1e-9))
    return {
        "wind_kw": levels_for("wind_kw", 100.0),
        "pv_kw": levels_for("pv_kw", 100.0),
        "diesel_units": tuple(range(diesel_max + 1)),
        "battery_kwh": levels_for("battery_kwh", 50.0),
        "pcs_kw": levels_for("pcs_kw", 50.0),
        "ups_kwh": levels_for("ups_kwh", 50.0),
        "ups_kw": levels_for("ups_kw", 50.0),
    }


def iter_capacity_grid(levels, cfg: ResilienceConfig, investment_budget_yuan: float,
                       required_ups_kwh: float, required_ups_kw: float):
    """Enumerate valid discrete capacities in nondecreasing investment cost.

    A heap over the multidimensional integer index grid avoids materialising
    the millions of possible combinations.  Every valid point below the
    incumbent investment bound is still reachable; no post-solution rounding
    is used.
    """
    names = ("wind_kw", "pv_kw", "diesel_units", "battery_kwh", "pcs_kw", "ups_kwh", "ups_kw")
    values = tuple(levels[name] for name in names)
    costs = (
        1500.0, 1000.0, 180000.0, 250.0, 400.0, 450.0, 250.0
    )
    # Every feasible design must meet the hard core UPS bridge requirement.
    # Shift those two coordinates to their smallest module indices before
    # traversing; this removes millions of provably invalid low-UPS points
    # while retaining every admissible module combination.
    lower = [0] * len(names)
    try:
        lower[names.index("ups_kwh")] = next(
            i for i, value in enumerate(values[names.index("ups_kwh")]) if value + 1e-9 >= required_ups_kwh
        )
        lower[names.index("ups_kw")] = next(
            i for i, value in enumerate(values[names.index("ups_kw")]) if value + 1e-9 >= required_ups_kw
        )
    except StopIteration as exc:
        raise ValueError("investment bound cannot buy the mandatory UPS bridge modules") from exc
    start = tuple(lower)
    start_cost = sum(costs[pos] * values[pos][idx] for pos, idx in enumerate(start))
    if start_cost > investment_budget_yuan + 1e-9:
        raise ValueError("investment bound is below the mandatory UPS bridge investment")
    heap = [(start_cost, start)]
    while heap:
        cost, index = heapq.heappop(heap)
        if cost > investment_budget_yuan + 1e-9:
            break
        data = {name: values[pos][idx] for pos, (name, idx) in enumerate(zip(names, index))}
        cap = Capacity(**data)
        if cap.ups_kwh + 1e-9 >= required_ups_kwh and cap.ups_kw + 1e-9 >= required_ups_kw:
            if cap.pcs_kw <= 1e-9 or cap.battery_kwh + 1e-9 >= cfg.battery_pcs_min_duration_h * cap.pcs_kw:
                yield cap
        # Canonical unique-parent traversal. The parent of a nonzero index is
        # obtained by decrementing its highest nonzero coordinate. A child can
        # increment only coordinates at or above the current highest nonzero
        # coordinate, visiting every bounded grid point exactly once.
        # Work in shifted coordinates so that the mandatory UPS floor is the
        # origin for uniqueness purposes.
        shifted = tuple(index[pos] - lower[pos] for pos in range(len(names)))
        highest = max((pos for pos, value in enumerate(shifted) if value), default=0)
        for pos in range(highest, len(names)):
            next_index = list(index)
            next_index[pos] += 1
            next_index = tuple(next_index)
            if next_index[pos] >= len(values[pos]):
                continue
            next_cost = cost + costs[pos] * (values[pos][next_index[pos]] - values[pos][index[pos]])
            if next_cost <= investment_budget_yuan + 1e-9:
                heapq.heappush(heap, (next_cost, next_index))


def assert_discrete_capacity_grid(capacities: list[Capacity]) -> None:
    if not capacities:
        raise ValueError("the discrete capacity grid is empty")
    for cap in capacities:
        assert_discrete_capacity(cap)


def find_feasible_reference_capacity(year, cfg: ResilienceConfig, scenarios, deadline=None, progress_callback=None):
    """Find a cost incumbent by adding diesel modules from zero upward.

    The returned capacity is only a branch-and-bound incumbent.  Its diesel
    count is discovered by feasibility checks and is never used as a planning
    upper bound.
    """
    base = {
        "wind_kw": 500.0,
        "pv_kw": 300.0,
        "battery_kwh": 1000.0,
        "pcs_kw": 200.0,
        # Must exceed the SOC-adjusted bridge requirement; this is only a
        # feasible incumbent seed, not a UPS planning upper bound.
        "ups_kwh": max(1000.0, 50.0 * math.ceil(required_ups_bridge_kwh(year, cfg) / 50.0)),
        "ups_kw": max(100.0, 50.0 * math.ceil(required_ups_bridge_kw(year, cfg) / 50.0)),
    }
    diesel_units = 0
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("planning deadline reached before a feasible seed was found")
        # If adding diesel alone cannot bridge a failure, increase storage in
        # this seed heuristic too. This is not a bound on the global domain.
        if diesel_units and diesel_units % 4 == 0:
            base["battery_kwh"] *= 2
            base["pcs_kw"] += 100.0
            base["ups_kwh"] += 500.0
        reference = Capacity(diesel_units=diesel_units, **base)
        evaluation = evaluate_capacity(year, reference, scenarios, cfg)
        if progress_callback:
            progress_callback({"phase": "reference_search", "reference": evaluation})
        if evaluation["feasible"]:
            return reference, evaluation
        diesel_units += 1


def planning_scenarios(year, cfg: ResilienceConfig, count: int, seed: int):
    """Keep the requested sample count while reserving explicit compound extremes."""
    compound_count = max(1, min(4, count // 4)) if count else 0
    joint_count = max(0, count - compound_count)
    scenarios = make_scenarios(year, cfg, joint_count, seed, "joint", scenario_id_start=0)
    scenarios.extend(make_scenarios(
        year, cfg, compound_count, seed + 500000, "compound_extreme",
        scenario_id_start=joint_count,
    ))
    return scenarios


def legacy_main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/synthetic_resilience_v2_fine_modules_day_ahead"))
    parser.add_argument("--hours", type=int, default=8760)
    parser.add_argument("--training-scenarios", type=int, default=16)
    parser.add_argument("--validation-scenarios", type=int, default=32)
    parser.add_argument("--time-limit-seconds", type=float, default=10000.0)
    parser.add_argument("--loss-of-load-cost", type=float, default=1000.0,
                        help="synthetic VOLL in yuan/kWh for core and rigid loss")
    args = parser.parse_args()
    if args.hours <= 0 or args.training_scenarios <= 0 or args.validation_scenarios <= 0:
        parser.error("hours and scenario counts must be positive")
    if args.loss_of_load_cost < 0 or not math.isfinite(args.loss_of_load_cost):
        parser.error("loss-of-load-cost must be finite and nonnegative")
    if args.time_limit_seconds < 0 or not math.isfinite(args.time_limit_seconds):
        parser.error("use 0 for unlimited or a positive finite time limit")
    if (args.output / "run_manifest.json").exists() or (args.output / "summary.json").exists():
        parser.error("output already contains a run; use a new directory to preserve results")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.time_limit_seconds if args.time_limit_seconds else None

    def write_json(path, value):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def progress(value):
        value = {**value, "total_elapsed_seconds": time.monotonic() - started, "pid": os.getpid()}
        write_json(args.output / "progress.json", value)
        selected = value.get("selected") or value.get("reference")
        if selected and selected["feasible"]:
            write_json(args.output / "incumbent.json", selected)
        print(json.dumps({key: val for key, val in value.items()
                          if key not in ("selected", "reference")}, ensure_ascii=False), flush=True)

    cfg = ResilienceConfig(
        hours=args.hours,
        training_scenarios=args.training_scenarios,
        validation_scenarios=args.validation_scenarios,
        loss_of_load_cost_yuan_per_kwh=args.loss_of_load_cost,
    )
    write_json(args.output / "resolved_config.json", asdict(cfg))
    source_paths = [Path(__file__).resolve(), Path(__file__).resolve().parents[1] /
                    "polar_reliability_planning/resilience_v2/model.py"]
    write_json(args.output / "run_manifest.json", {
        "pid": os.getpid(), "synthetic": True, "module_sizes": CAPACITY_STEPS,
        "code_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "time_limit_seconds": args.time_limit_seconds,
        "time_limit_scope": "reference search and capacity search; final validation timed separately",
        "forecast": "synthetic perfect next-calendar-day Boolean at 00:00",
        "cost_basis": "capital investment plus one simulated year of expected operation and loss costs",
        "scenario_weight_basis": "equal stress-sample weights; not calibrated event probabilities",
        "policy": "fixed causal reserve/start/priority-dispatch rule; capacity decisions optimized",
    })
    year = generate_synthetic_year(args.hours, cfg.seed)
    training = planning_scenarios(year, cfg, cfg.training_scenarios, cfg.seed + 1)
    # This incumbent is discovered by starting at zero diesel units and adding
    # one unit only when the current reference is infeasible.  It is not a
    # diesel-unit limit: the grid below starts at zero and adds units until the
    # incumbent investment cost is reached.
    try:
        reference_capacity, reference_evaluation = find_feasible_reference_capacity(
            year, cfg, training, deadline, progress)
    except TimeoutError as exc:
        progress({"phase": "finished", "status": "time_limit_without_feasible_incumbent",
                  "planning_complete": False, "reason": str(exc)})
        write_json(args.output / "summary.json", {
            "status": "time_limit_without_feasible_incumbent", "selected": None,
            "validation": None, "metadata": {"planning_complete": False,
            "planning_elapsed_seconds": time.monotonic() - started, "synthetic": True}})
        return 2
    levels, capacities = global_capacity_grid_iterator(year, cfg, reference_evaluation["objective_yuan"])
    # The iterator is ordered by investment and checked point-by-point; no
    # candidate is rounded after solving.  The full grid count below counts
    # declared integer index points before budget/coupling pruning.
    declared_count = 1
    for values in levels.values():
        declared_count *= len(values)
    plan = plan_capacity(
        year, cfg, capacities, training,
        max(0.0, deadline - time.monotonic()) if deadline is not None else None,
        reference_capacity, declared_candidate_count=declared_count,
        investment_ordered=True, progress_callback=progress,
    )
    plan.metadata["reference_and_search_elapsed_seconds"] = time.monotonic() - started
    plan.metadata["requested_time_limit_seconds"] = args.time_limit_seconds
    plan.metadata["module_sizes"] = CAPACITY_STEPS
    plan.metadata["resolved_config"] = asdict(cfg)
    plan.metadata["capacity_grid"] = {
        "wind_kw": list(levels["wind_kw"]),
        "pv_kw": list(levels["pv_kw"]),
        "diesel_units": list(levels["diesel_units"]),
        "battery_kwh": list(levels["battery_kwh"]),
        "pcs_kw": list(levels["pcs_kw"]),
        "ups_kwh": list(levels["ups_kwh"]),
        "ups_kw": list(levels["ups_kw"]),
    }
    plan.metadata["capacity_search_rule"] = (
        "all nonnegative integer module combinations; only valid economic incumbent, "
        "physical UPS bridge and battery/PCS coupling bounds remove candidates"
    )
    plan.metadata["all_capacity_variables_discrete"] = True
    plan.metadata["post_solution_rounding"] = False
    plan.metadata["reference_capacity"] = reference_evaluation["capacity"]
    plan.metadata["reference_incumbent_objective_yuan"] = reference_evaluation["objective_yuan"]
    plan.metadata["training_scenario_count"] = len(training)
    plan.metadata["training_compound_extreme_count"] = sum(s.compound_extreme for s in training)
    plan.metadata["loss_of_load_cost_yuan_per_kwh"] = cfg.loss_of_load_cost_yuan_per_kwh
    plan.metadata["flex_adjustment_cost_yuan_per_kwh"] = cfg.flex_adjustment_cost_yuan_per_kwh
    plan.metadata["weather_forecast_information"] = (
        "calendar-day Boolean released during day d for whether day d+1 is extreme; "
        "current weather is observed, future faults are not revealed"
    )
    validation_started = time.monotonic()
    validation = None
    selected_trace = None
    if plan.selected is not None:
        progress({"phase": "validation", "selected": plan.selected,
                  "status": plan.status, "planning_complete": plan.metadata["planning_complete"]})
        selected = Capacity(**plan.selected["capacity"])
        assert_discrete_capacity(selected)
        holdout = planning_scenarios(year, cfg, cfg.validation_scenarios, cfg.seed + 10001)
        validation = validate_capacity(year, selected, cfg, holdout)
        validation["scenario_class_metrics"] = {}
        for risk_class in ("weather", "random", "joint", "compound_extreme"):
            count = max(4, min(16, cfg.validation_scenarios))
            scenarios = make_scenarios(year, cfg, count, cfg.seed + 20000 + len(risk_class), risk_class)
            validation["scenario_class_metrics"][risk_class] = evaluate_capacity(year, selected, scenarios, cfg)
        worst_scenario = max(
            holdout,
            key=lambda scenario: sum(simulate(year, selected, scenario, cfg).metrics[key]
                                     for key in ("core_unserved_kwh", "regular_unserved_kwh")),
        )
        selected_trace = simulate(year, selected, worst_scenario, cfg, trace=True).trace
        plan.validation = validation
        plan.metadata["validation_scenario_count"] = len(holdout)
        plan.metadata["validation_compound_extreme_count"] = sum(s.compound_extreme for s in holdout)
    plan.metadata["validation_elapsed_seconds"] = time.monotonic() - validation_started
    plan.metadata["total_elapsed_seconds"] = time.monotonic() - started
    write_plan_outputs(args.output, year, plan, validation, selected_trace)
    progress({"phase": "finished", "status": plan.status, "selected": plan.selected,
              "planning_complete": plan.metadata["planning_complete"],
              "validation_feasible": validation["feasible"] if validation else None})
    # Keep the generated grid/renewable-bus outage paths inspectable without
    # expanding the main hourly data file by hundreds of thousands of rows.
    with (args.output / "scenario_fault_events.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["role", "risk_class", "scenario_id", "seed", "event", "start_hour", "end_hour"])
        writer.writeheader()
        for role, scenarios in (("training", training), ("validation", planning_scenarios(year, cfg, cfg.validation_scenarios, cfg.seed + 10001))):
            for scenario in scenarios:
                for event_name, values in (("grid_bus_fault", scenario.grid_fault), ("renewable_bus_fault", scenario.renewable_bus_fault)):
                    start = None
                    for hour, active in enumerate(values.tolist() + [False]):
                        if active and start is None:
                            start = hour
                        elif not active and start is not None:
                            writer.writerow({"role": role, "risk_class": scenario.risk_class, "scenario_id": scenario.scenario_id,
                                             "seed": scenario.seed, "event": event_name, "start_hour": start, "end_hour": hour - 1})
                            start = None
    print(json.dumps({
        "status": plan.status,
        "output": str(args.output.resolve()),
        "selected": plan.selected["capacity"] if plan.selected else None,
        "training_risk": plan.selected["regular_risk"] if plan.selected else None,
        "validation_risk": validation["regular_risk"] if validation else None,
        "validation_core_unserved_kwh": validation["core_risk"] if validation else None,
        "all_pcs_grid_forming": cfg.all_pcs_grid_forming,
        "all_diesel_grid_forming": cfg.all_diesel_grid_forming,
    }, ensure_ascii=False, indent=2))
    return 0 if plan.selected else 2


def main() -> int:
    # Existing users of this entry point get the new joint MILP by default.
    # Explicit opt-in is required to reproduce the historical fixed-policy
    # enumeration; it is not the current planning method.
    if "--legacy-enumeration" in sys.argv:
        sys.argv.remove("--legacy-enumeration")
        return legacy_main()
    from scripts.run_resilience_milp import main as milp_main
    return milp_main()


if __name__ == "__main__":
    raise SystemExit(main())
