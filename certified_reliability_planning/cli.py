"""Separate CLI and evidence output for the certified planning program."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
from itertools import product
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import tomllib

import numpy as np

from .planner import CertifiedPlanner, PlannerOptions

ROOT = Path(__file__).resolve().parent.parent


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="独立规划程序：单调容量搜索、UC 区间与总体风险证书")
    parser.add_argument("command", choices=("plan", "demo", "validate-small", "verify-isolation"))
    parser.add_argument("--config", type=Path, default=ROOT / "config/certified_zhongshan.toml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--hours", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--epsilon-cost", type=float, help="绝对成本容差，元/所选时域")
    parser.add_argument("--mip-gap", "--relative-gap", dest="relative_gap", type=float,
                        help="经济求解和最终证书的相对 gap，0.01 表示 1%%；替代绝对成本停止条件")
    parser.add_argument("--eens-limit", type=float)
    parser.add_argument("--cvar-limit", type=float)
    parser.add_argument("--alpha", type=float)
    for name in ("initial-samples", "max-samples", "max-stages", "max-oracle-calls", "max-economic-calls"):
        parser.add_argument("--" + name, type=int)
    time_budget = parser.add_mutually_exclusive_group()
    time_budget.add_argument("--max-seconds", type=float,
                             help="总规划时间预算（秒），0 表示不设置总时间上限")
    time_budget.add_argument("--no-time-limit", dest="max_seconds", action="store_const", const=0.0,
                             help="取消总规划时间上限；保留单次求解和其他计算预算")
    for name in ("base-call-seconds", "max-call-seconds"):
        parser.add_argument("--" + name, type=float)
    return parser.parse_args(argv)


def options_from(args, values=None):
    values = dict(values or {})
    for name in PlannerOptions.__dataclass_fields__:
        if getattr(args, name, None) is not None:
            values[name] = getattr(args, name)
    return PlannerOptions(**values)


def verify_isolation():
    manifest = json.loads((ROOT / "certified_reliability_planning/legacy_sha256.json").read_text())
    changed = [p for p, digest in manifest.items()
               if not (ROOT / p).is_file() or hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != digest]
    return {"passed": not changed, "checked_existing_files": len(manifest), "changed_or_missing": changed,
            "scope": "tracked files present before the independent planner was added"}


def _event_writer(output):
    last_update = 0.0

    def record(event):
        nonlocal last_update
        with (output / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(event), ensure_ascii=False, allow_nan=False) + "\n")
        now = time.monotonic()
        if event["action"] in ("stage", "risk_certificate") or now - last_update >= 5:
            write_json(output / "progress.json", {"status": "running", **event})
            print(f"[{datetime.now():%H:%M:%S}] {event['action']} "
                  f"m={event.get('sample_count', '-')} point={event.get('point', '-')} "
                  f"elapsed={event['elapsed_seconds']:.1f}s", flush=True)
            last_update = now
    return record


def run_demo(args, output):
    from .benchmarks import tiny_benchmark
    oracle = tiny_benchmark(seed=7 if args.seed is None else args.seed)
    options = options_from(args, {"epsilon_cost": 0.01, "max_samples": 4096, "max_stages": 12,
                                  "max_oracle_calls": 30000, "max_economic_calls": 500,
                                  "max_seconds": 120.0})
    eens = 0.3 if args.eens_limit is None else args.eens_limit
    cvar = 0.5 if args.cvar_limit is None else args.cvar_limit
    alpha = 0.5 if args.alpha is None else args.alpha
    planner = CertifiedPlanner(oracle.candidates, oracle.cost_lower_bounds, oracle,
        oracle.operation, oracle.economic, options, eens, cvar, alpha, _event_writer(output))
    write_json(output / "resolved_config.json", {"benchmark": "analytic_two_state_nine_capacity",
        "seed": oracle.seed, "options": asdict(options), "loss_bound": oracle.loss_bound,
        "limits": {"eens": eens, "cvar": cvar, "alpha": alpha}})
    result = planner.run()
    truth = oracle.truth_table(eens, cvar, alpha)
    feasible = [row for row in truth if row["feasible"]]
    optimum = min(feasible, key=lambda row: row["cost"]) if feasible else None
    checks = []
    for witness in result["risk_witnesses"]:
        p = witness["point"]
        for row in truth:
            if witness["label"] == "infeasible" and all(a <= b for a, b in zip(row["point"], p)):
                checks.append(not row["feasible"])
            if witness["label"] == "feasible" and all(a >= b for a, b in zip(row["point"], p)):
                checks.append(row["feasible"])
    valid = all(checks)
    if result["status"] == "certified_optimal":
        exact = next(row for row in truth if row["point"] == result["incumbent"])
        valid &= exact["feasible"] and exact["cost"] <= optimum["cost"] + options.cost_tolerance(result["upper_bound_cost"])
        result["incumbent_population_truth"] = exact
    elif result["status"] == "certified_infeasible":
        valid &= optimum is None
    result.update(problem="analytic_demo", seed=oracle.seed, truth_optimum=optimum,
                  validation={"certificate_matches_enumeration": bool(valid),
                              "all_propagated_labels_checked": len(checks),
                              "population_law": "two equiprobable demand states; truth never used by search"})
    write_json(output / "population_truth.json", truth)
    np.save(output / "iid_uniform_samples.npy", oracle.draws)
    planner.save_state(output / "certificate_state.npz")
    if args.command == "validate-small":
        result["validation"]["passed"] = bool(valid and result["certificate"] is not None)
    return result


def run_plan(args, output):
    import gurobipy as gp
    from polar_reliability_planning.config import SolverOptions, UnitCommitmentOptions
    from polar_reliability_planning.data import COMPONENTS, load_case
    from .oracles import MicrogridOracle
    from .sampling import SharedScenarioStream
    from .economic_search import EconomicSearch

    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    system_path = Path(config.get("system_config", "system_config.toml"))
    if not system_path.is_absolute():
        system_path = args.config.resolve().parent / system_path
    with system_path.open("rb") as handle:
        system = tomllib.load(handle)
    options = options_from(args, config.get("certification"))
    solver = SolverOptions(**config.get("solver", {}))
    if options.relative_gap is not None:
        solver = replace(solver, mip_gap=options.relative_gap)
    data = load_case(args.data_root or system["data_root"], args.case or system["case"],
        start_hour=system.get("start_hour", 0), hours=args.hours if args.hours is not None else system.get("hours", 8760),
        modules=system.get("modules"), load_scale=system.get("load_scale", 1.0),
        unit_commitment=UnitCommitmentOptions(**system.get("unit_commitment", {})),
        max_units=system.get("max_units"))
    reliability = system.get("reliability", {})
    seed = args.seed if args.seed is not None else config.get("seed", 20260910)
    weather = system.get("weather", {})
    stream = SharedScenarioStream(data, seed, system.get("failures", {}), weather)
    cardinality = math.prod(hi - lo + 1 for lo, hi in data.unit_bounds.values())
    max_designs = config.get("max_designs", 2000000)
    if cardinality > max_designs:
        raise ValueError(f"Declared grid has {cardinality} points, exceeding max_designs={max_designs}; "
                         "increase that explicit memory limit or configure a smaller grid")
    candidates = np.asarray(list(product(*(range(data.unit_bounds[k][0], data.unit_bounds[k][1] + 1)
                                           for k in COMPONENTS))), dtype=np.int64)
    cost_lower = candidates @ np.asarray([data.period_cost_per_unit[k] for k in COMPONENTS])
    # The initial capital expression is a lower bound; round towards zero.
    cost_lower = np.maximum(0, np.nextafter(cost_lower, -np.inf))
    eens = args.eens_limit if args.eens_limit is not None else reliability.get("eens_limit_kwh", 100.0)
    cvar = args.cvar_limit if args.cvar_limit is not None else reliability.get("cvar_limit_kwh", 1000.0)
    alpha = args.alpha if args.alpha is not None else reliability.get("cvar_alpha", 0.95)
    resolved = {"system_config": str(system_path.resolve()), "system_config_sha256": hashlib.sha256(system_path.read_bytes()).hexdigest(),
                "system": system, "case": data.name, "hours": data.hours, "dt_hours": data.dt_hours,
                "unit_bounds": data.unit_bounds, "module_sizes": data.module_sizes,
                "input_manifest": data.manifest, "input_metadata": data.input_metadata,
                "unit_commitment": asdict(data.unit_commitment), "options": asdict(options),
                "solver": asdict(solver), "limits": {"eens": eens, "cvar": cvar, "alpha": alpha},
                "stream": stream.metadata, "cardinality": cardinality,
                "cost_basis": "period_capital_plus_nominal_fuel_and_startup_cost",
                "information_structure": "perfect_foresight_UC_with_free_cyclic_initial_usable_energy",
                "numerical_scope": "conditional_on_Gurobi_bounds_and_dispatch_audits_with_declared_tolerances"}
    write_json(output / "resolved_config.json", resolved)
    gap_text = (f"relative_gap={options.relative_gap:.2%}" if options.relative_gap is not None
                else f"epsilon={options.epsilon_cost}元")
    print(f"{data.name}: {data.hours}h, K={cardinality}, B={stream.loss_bound:.6f}kWh, "
          f"delta={options.delta}, {gap_text}", flush=True)
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        oracle = MicrogridOracle(data, solver, env)
        region = EconomicSearch(data, solver, env)
        units = lambda p: dict(zip(COMPONENTS, map(int, p)))
        planner = CertifiedPlanner(candidates, cost_lower, stream,
            lambda p, s, b, g: oracle.operation(units(p), stream.scenario(units(p), s), b, g),
            lambda p, b, g: oracle.economic(units(p), b, g), options, eens, cvar, alpha,
            _event_writer(output), region_search=region)
        try:
            result = planner.run()
            planner.save_state(output / "certificate_state.npz")
            if stream.samples:
                stream.save(output / "shared_scenarios.npz")
            if result["incumbent"] is not None:
                point = tuple(result["incumbent"])
                result["units"] = units(point)
                result["capacities"] = data.capacities(units(point))
                if point in planner.economic_dispatches:
                    np.savez_compressed(output / "nominal_dispatch.npz", **planner.economic_dispatches[point])
            if result["best_nominal_candidate"] is not None:
                candidate = result["best_nominal_candidate"]
                point = tuple(candidate["point"])
                candidate["units"] = units(point)
                candidate["capacities"] = data.capacities(units(point))
                if point in planner.economic_dispatches:
                    np.savez_compressed(output / "candidate_dispatch.npz", **planner.economic_dispatches[point])
            result.update(problem="microgrid", case=data.name, hours=data.hours, seed=seed,
                          scenario_fingerprint=stream.fingerprint,
                          information_structure=resolved["information_structure"],
                          numerical_scope=resolved["numerical_scope"],
                          population_scope="specified_weather_failure_model; not field_parameter_uncertainty",
                          independent_holdout="not_run; training certificate uses simultaneous population bounds")
            return result
        finally:
            region.close()
            oracle.close()


def main(argv=None):
    args = parse_args(argv)
    if args.command == "verify-isolation":
        result = verify_isolation()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1
    output = args.output or ROOT / "outputs/certified" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        print(f"错误：结果目录必须为空，拒绝覆盖 {output}", file=sys.stderr)
        return 1
    output.mkdir(parents=True, exist_ok=True)
    print(f"独立结果目录：{output}", flush=True)
    try:
        result = run_plan(args, output) if args.command == "plan" else run_demo(args, output)
        result["legacy_isolation"] = verify_isolation()
        write_json(output / "summary.json", result)
        write_json(output / "progress.json", {"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]})
        print(f"状态：{result['status']}; 容量={result['incumbent']}; "
              f"成本区间=[{result['lower_bound_cost']}, {result['upper_bound_cost']}]", flush=True)
        print(f"结果：{output / 'summary.json'}", flush=True)
        if not result["legacy_isolation"]["passed"]:
            return 1
        if args.command == "validate-small" and not result["validation"]["passed"]:
            return 1
        return {"certified_optimal": 0, "certified_infeasible": 2, "budget_exhausted": 4}[result["status"]]
    except Exception as error:
        # Also persist solver/license errors; KeyboardInterrupt/SystemExit are
        # intentionally not caught by this ordinary-Exception boundary.
        write_json(output / "summary.json", {"status": "error", "error_type": type(error).__name__,
                                             "error": str(error)})
        write_json(output / "progress.json", {"status": "error", "error": str(error)})
        print(f"错误：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
