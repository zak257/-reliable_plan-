from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import gurobipy as gp

from .config import ROOT, ReliabilityLimits, SolverOptions, UnitCommitmentOptions, read_config
from .data import COMPONENTS, load_case
from .planning import MasterMILP, optimize_reliability
from .reliability import ReliabilityOracle
from .reliability.operation_model import OptimizationError
from .reporting import write_csv, write_dispatch, write_json
from .scenario_generation import ScenarioPool, generate_pool
from .validation.monte_carlo_validation import audit_nominal_solution, validate_capacity
from .validation.sensitivity_analysis import run_sensitivity
from .validation.small_system import validate_small_system


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="cap_plan 数据 + Gurobi：可靠性边界识别容量规划")
    parser.add_argument("command", nargs="?", default="plan", choices=("plan", "baseline", "evaluate", "validate-small", "sensitivity"))
    parser.add_argument("--config", type=Path, default=ROOT / "config/system_config.toml")
    parser.add_argument("--solver-config", type=Path, default=ROOT / "config/solver_config.toml")
    parser.add_argument("--case")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--hours", type=int)
    parser.add_argument("--start-hour", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument("--eens-limit", type=float, help="kWh over the selected horizon; not annualized for short runs")
    parser.add_argument("--cvar-limit", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--climate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lift-cuts", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--oracle-time-limit", type=float)
    parser.add_argument("--unit-commitment", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mip-gap", type=float)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--solver-log", action="store_true")
    parser.add_argument("--scenarios", type=Path, help="Reuse an existing training_scenarios.npz")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--units", help="JSON object (or JSON file) with integer module counts")
    group.add_argument("--capacity", help="JSON object (or JSON file) with capacities in kW / kWh")
    parser.add_argument("--eens-limits", help="Comma-separated kWh limits for sensitivity")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _read_units(args, data):
    source = args.units or args.capacity
    if source is None:
        raise ValueError("evaluate requires --units or --capacity")
    raw = json.loads(source if source.lstrip().startswith("{") else Path(source).read_text(encoding="utf-8"))
    if args.capacity:
        if set(raw) != set(COMPONENTS):
            raise ValueError(f"capacity must contain exactly {COMPONENTS}")
        raw = {k: raw[k] / data.module_sizes[k] for k in COMPONENTS}
    return data.validate_units(raw)


def run(args, output: Path) -> int:
    if args.command != "evaluate" and (args.units is not None or args.capacity is not None):
        raise ValueError("--units and --capacity are only used by evaluate")
    config = read_config(args.config)
    options = SolverOptions(**read_config(args.solver_config))
    overrides = {k: getattr(args, k) for k in ("time_limit", "oracle_time_limit", "mip_gap", "threads") if getattr(args, k) is not None}
    if args.solver_log:
        overrides["output_flag"] = True
    options = replace(options, **overrides)
    commitment = UnitCommitmentOptions(**config.get("unit_commitment", {}))
    if args.unit_commitment is not None:
        commitment = replace(commitment, enabled=args.unit_commitment)
    history = []

    def on_iteration(row):
        history.append(row)
        write_json(output / "history.json", history)
        write_csv(output / "history.csv", history)
        relation = ">=" if row.get("metrics_are_lower_bounds") else "="
        _log(f"轮次 {row['iteration']}: cost={row['objective_yuan']:.3f} 元, EENS{relation}{row['eens_kwh']:.6f} kWh, "
             f"CVaR{relation}{row['cvar_kwh']:.6f} kWh, feasible={row['feasible']}, n={row['units']}")

    _log(f"Python: {sys.executable}; Gurobi: {gp.gurobi.version()}; 输出: {output}")
    with gp.Env(params={"OutputFlag": int(options.output_flag)}) as env:
        if args.command == "validate-small":
            result = validate_small_system(env, on_iteration, unit_commitment=commitment.enabled)
            write_json(output / "summary.json", result)
            write_csv(output / "grid.csv", result["grid"]["rows"])
            _log(f"48点穷举验证通过；目标值差={result['objective_difference_yuan']:.3g} 元；"
                 f"可靠性cut={len(result['boundary']['cuts'])}")
            return 0
        case = args.case or config["case"]
        data = load_case(args.data_root or config["data_root"], case,
                         args.start_hour if args.start_hour is not None else config["start_hour"],
                         args.hours if args.hours is not None else config["hours"], config["modules"], config.get("load_scale", 1.0), commitment)
        reliability_config = dict(config["reliability"])
        for key in ("samples", "seed", "validation_samples", "validation_seed"):
            if getattr(args, key) is not None:
                reliability_config[key] = getattr(args, key)
        if reliability_config["validation_samples"] < 0:
            raise ValueError("validation_samples cannot be negative")
        weather = dict(config.get("weather", {}))
        if args.climate is not None:
            weather["enabled"] = args.climate
        eens_limit = args.eens_limit if args.eens_limit is not None else reliability_config.get("eens_limit_kwh", data.demand_kwh * reliability_config.get("eens_fraction", 0.001))
        limits = ReliabilityLimits(eens_limit, args.cvar_limit if args.cvar_limit is not None else reliability_config.get("cvar_limit_kwh"),
                                   args.alpha if args.alpha is not None else reliability_config.get("cvar_alpha", 0.95),
                                   reliability_config.get("comparison_tolerance_kwh", 1e-5))
        planning = config["planning"]
        max_iterations = args.max_iterations if args.max_iterations is not None else planning["max_iterations"]
        lift_cuts = args.lift_cuts if args.lift_cuts is not None else planning.get("lift_cuts", False)
        write_json(output / "input_manifest.json", {"files": data.manifest, "metadata": data.input_metadata})
        resolved = {"command": args.command, "python": sys.executable, "gurobi_version": gp.gurobi.version(),
                    "case": data.name, "hours": data.hours, "demand_kwh": data.demand_kwh,
                    "unit_bounds": data.unit_bounds, "module_sizes": data.module_sizes,
                    "annual_cost_per_unit": data.annual_cost_per_unit, "solver": asdict(options), "limits": asdict(limits),
                    "unit_commitment": asdict(commitment),
                    "reliability": reliability_config, "weather": weather, "failures": config.get("failures", {}),
                    "planning": {"max_iterations": max_iterations, "lift_cuts": lift_cuts},
                    "source_config": str(args.config.resolve()), "scenario_source": str(args.scenarios.resolve()) if args.scenarios else None}
        write_json(output / "resolved_config.json", resolved)
        _log(f"{data.name}: {data.hours} 小时, 用电量 {data.demand_kwh:.3f} kWh, EENS限值 {limits.eens_kwh:.3f} kWh, "
             f"CVaR({limits.alpha})限值={limits.cvar_kwh} kWh, UC={commitment.enabled}")
        base = {"case": data.name, "hours": data.hours, "demand_kwh": data.demand_kwh, "limits": asdict(limits),
                "cost_basis": "straight_line_annual_capital_times_hours_over_8760_plus_nominal_fuel_and_startups",
                "unit_commitment": asdict(commitment),
                "reliability_basis": "perfect_foresight_minimum_unserved_energy_on_fixed_samples"}
        if args.command == "baseline":
            master = MasterMILP(data, options, env=env)
            try:
                solution = master.solve()
                audit = audit_nominal_solution(data, solution, options, env)
                write_dispatch(output / "dispatch.csv", data, solution)
                write_json(output / "summary.json", {**base, "status": "nominal_baseline", "solution": solution.summary(), "audit": audit})
                _log(f"基线求解完成: cost={solution.objective_yuan:.3f} 元, n={solution.units}, audit={audit}")
            finally:
                master.close()
            return 0
        _log("读取固定故障样本" if args.scenarios else "生成固定故障样本")
        pool = ScenarioPool.load(args.scenarios) if args.scenarios else generate_pool(data, reliability_config["samples"],
                   reliability_config["seed"], config.get("failures", {}), weather)
        pool.check_data(data)
        if args.command == "plan" and reliability_config["validation_samples"]:
            if not {"seed", "failures", "weather"}.issubset(pool.metadata):
                raise ValueError("Loaded pool lacks its generating law; use --validation-samples 0 for a custom finite distribution")
            if reliability_config["validation_seed"] == pool.metadata["seed"]:
                raise ValueError("Independent validation must use a different seed from training")
        pool.save(output / "training_scenarios.npz")
        resolved["actual_training_pool"] = {"samples": pool.samples, "fingerprint": pool.fingerprint, "metadata": pool.metadata}
        write_json(output / "resolved_config.json", resolved)
        _log(f"样本数={pool.samples}, SHA256={pool.fingerprint[:16]}…")
        if args.command == "sensitivity":
            if not args.eens_limits:
                raise ValueError("sensitivity requires --eens-limits")
            values = [float(v) for v in args.eens_limits.split(",")]
            rows = run_sensitivity(data, pool, limits, values, options, max_iterations, lift_cuts, env, on_iteration)
            write_json(output / "summary.json", {**base, "sensitivity": rows, "sample_fingerprint": pool.fingerprint})
            write_csv(output / "sensitivity.csv", rows)
            return 0 if all(r["solution"] is not None for r in rows) else 2
        def oracle_progress(done, total, eens, cvar):
            _log(f"Oracle {done}/{total}: 固定样本 EENS下界={eens:.6f}, CVaR下界={cvar:.6f} kWh")

        oracle = ReliabilityOracle(data, pool, limits, options, env, on_progress=oracle_progress)
        try:
            if args.command == "evaluate":
                units = _read_units(args, data)
                result = oracle.evaluate(units)
                write_json(output / "summary.json", {**base, "units": units, "capacities": data.capacities(units),
                                                     "reliability": result.summary()})
                write_csv(output / "scenario_losses.csv", [{"scenario": s, "probability": pool.probabilities[s], "loss_kwh": q}
                                                           for s, q in enumerate(result.losses_kwh)])
                _log(f"EENS={result.eens_kwh:.6f}, CVaR={result.cvar_kwh:.6f}, feasible={result.feasible}")
                return 0
            result = optimize_reliability(data, oracle, options, max_iterations, lift_cuts, on_iteration, env)
            summary = {**base, "status": result.status, "elapsed_seconds": result.elapsed_seconds,
                       "solution": result.solution.summary() if result.solution else None,
                       "reliability": result.reliability.summary() if result.reliability else None,
                       "lower_bound_yuan": result.lower_bound_yuan, "iterations": len(result.history),
                       "cuts": [c.as_dict() for c in result.cuts], "oracle_evaluations": len(oracle.cache),
                       "oracle_scenario_solves": oracle.operation.solve_count,
                       "oracle_relaxation_lp_solves": oracle.relaxation_oracle.operation.solve_count if oracle.relaxation_oracle else 0,
                       "oracle_full_mip_solves": oracle.operation.full_mip_solves,
                       "oracle_zero_loss_certificates": oracle.operation.zero_loss_certificates}
            write_json(output / "summary.json", summary)
            write_csv(output / "oracle_evaluations.csv", [{"units": dict(zip(COMPONENTS, key)), **v.summary()}
                                                         for key, v in oracle.cache.items()])
            if result.solution is None:
                _log(f"规划停止: {result.status}；没有输出已通过可靠性约束的容量方案")
                return 2
            write_dispatch(output / "dispatch.csv", data, result.solution)
            write_csv(output / "scenario_losses.csv", [{"scenario": s, "probability": pool.probabilities[s], "loss_kwh": q}
                                                       for s, q in enumerate(result.reliability.losses_kwh)])
            summary["audit"] = audit_nominal_solution(data, result.solution, options, env)
            count = reliability_config["validation_samples"]
            if count:
                validation_seed = reliability_config["validation_seed"]
                _log(f"独立验证: {count} 个样本, seed={validation_seed}")
                # Loaded training pools carry their actual failure/weather law.
                validation_pool = generate_pool(data, count, validation_seed,
                    pool.metadata.get("failures", config.get("failures", {})), pool.metadata.get("weather", weather))
                validation_pool.save(output / "validation_scenarios.npz")
                validation = validate_capacity(data, result.solution.units, validation_pool, limits, options, env)
                summary["validation"] = {k: v for k, v in validation.items() if k != "losses_kwh"}
                write_csv(output / "validation_losses.csv", [{"scenario": s, "loss_kwh": q}
                                                             for s, q in enumerate(validation["losses_kwh"])])
                summary["validation_status"] = "holdout_passed" if validation["feasible"] else "holdout_failed"
                _log(f"独立验证 EENS={validation['eens_kwh']:.6f} kWh, CVaR={validation['cvar_kwh']:.6f} kWh, passed={validation['feasible']}")
            else:
                summary["validation_status"] = "not_requested"
            write_json(output / "summary.json", summary)
            _log(f"完成: {result.status}; cost={result.solution.objective_yuan:.3f} 元; capacities={result.solution.capacities}")
            return 0 if summary["validation_status"] != "holdout_failed" else 3
        finally:
            oracle.close()


def main(argv=None) -> int:
    args = parse_args(argv)
    output = (args.output or ROOT / "outputs" / f"{args.command}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}").resolve()
    # Protect prior research results against accidental overwrite.
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        print(f"ERROR: output directory is nonempty: {output}; use a new --output directory", file=sys.stderr)
        return 1
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        return run(args, output)
    except (ValueError, KeyError, OSError, OptimizationError, gp.GurobiError) as exc:
        write_json(output / "error.json", {"error_type": type(exc).__name__, "message": str(exc),
                                          "elapsed_seconds": time.perf_counter() - started})
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
