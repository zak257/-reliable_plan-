"""Exact finite library/grid search with valid nonnegative investment bounds.

No legacy monotonicity/dominance cuts. A failed policy excludes only that
capacity/policy pair in the declared library. Checkpoints store completed paths;
an interrupted path is deterministically replayed from its keyed primitives.
"""
from dataclasses import asdict
import itertools
import json
import time

from ..data.cap_plan_loader import COMPONENTS
from ..reliability.causal_simulator import simulate
from ..reliability.recovery_policy import RecoveryPolicy
from ..reliability.risk_bounds import partial_bounds, passes, risk_summary
from ..scenario_generation.primitive_noise import PrimitiveNoise


class PathArchive:
    def __init__(self, path):
        self.path = path
        self.records = {}
        if path.exists():
            valid_end = 0
            with path.open("rb") as stream:
                lines = stream.readlines()
            for i, raw in enumerate(lines):
                try:
                    value = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    if i != len(lines) - 1:
                        raise ValueError("Corrupt checkpoint before final record")
                    with path.open("r+b") as stream:
                        stream.truncate(valid_end)
                    break
                self.records[value["key"]] = value
                valid_end += len(raw)

    def save(self, record):
        self.records[record["key"]] = record
        with self.path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()


def evaluate_policy(data, weather, units, policy, settings, reliability, archive,
                    role="training", early_screen=True, noise_cache=None):
    samples = reliability["samples"] if role == "training" else reliability["validation_samples"]
    seed = reliability["seed"] if role == "training" else reliability["validation_seed"]
    records = []
    key_prefix = f"{role}:{tuple(units[k] for k in COMPONENTS)}:{policy.fingerprint}:"
    maximum = {k: hi for k, (_, hi) in data.unit_bounds.items()}
    for scenario in range(samples):
        key = key_prefix + str(scenario)
        if key in archive.records:
            record = archive.records[key]
        else:
            noise_key = (seed, scenario)
            noise = (noise_cache or {}).get(noise_key)
            if noise is None:
                noise = PrimitiveNoise(seed, scenario, data.hours, maximum, settings.warmup_hours)
                if noise_cache is not None:
                    # A bounded cache is an exact RNG reuse, not a state approximation.
                    if len(noise_cache) >= 16:
                        noise_cache.pop(next(iter(noise_cache)))
                    noise_cache[noise_key] = noise
            result = simulate(data, weather, units, policy, settings, noise)
            record = dict(key=key, role=role, scenario=scenario, seed=seed, units=units,
                          policy=policy.name, valid=result.valid, loss_kwh=result.loss_kwh,
                          operating_cost_yuan=result.operating_cost_yuan, metrics=result.metrics, error=result.error)
            archive.save(record)
        records.append(record)
        if not record["valid"]:
            return dict(status="policy_execution_failed", feasible=False, complete=False, records=records,
                        error=record["error"], scope="this_capacity_and_policy_only")
        if early_screen and len(records) < samples:
            lower, upper = partial_bounds([r["loss_kwh"] for r in records], samples, data.demand_kwh, reliability["alpha"])
            if not passes(lower, reliability):
                return dict(status="policy_risk_lower_bound_failed", feasible=False, complete=False,
                            risk_lower_bound=lower, risk_upper_bound=upper, records=records,
                            scope="this_capacity_and_policy_only")
    risk = risk_summary([r["loss_kwh"] for r in records], reliability["alpha"])
    investment = sum(data.period_cost_per_unit[k] * units[k] for k in COMPONENTS)
    return dict(status="evaluated", feasible=passes(risk, reliability), complete=True, risk=risk,
                investment_cost_yuan=investment,
                expected_operating_cost_yuan=sum(r["operating_cost_yuan"] for r in records) / samples,
                objective_yuan=investment + sum(r["operating_cost_yuan"] for r in records) / samples,
                records=records)


def plan_grid(data, weather, config, settings, output, max_designs=None, progress=None):
    policies = [RecoveryPolicy(**p) for p in config["policies"]]
    grid = config["grid"]
    if set(grid) != set(COMPONENTS):
        raise ValueError(f"Grid must declare {COMPONENTS}")
    for component in COMPONENTS:
        if not grid[component] or len(grid[component]) != len(set(grid[component])):
            raise ValueError("Grid axes must be nonempty with unique integer entries")
    designs = [data.validate_units(dict(zip(COMPONENTS, values))) for values in itertools.product(*(grid[k] for k in COMPONENTS))]
    investment = lambda x: sum(data.period_cost_per_unit[k] * x[k] for k in COMPONENTS)
    designs.sort(key=lambda x: (investment(x), tuple(x[k] for k in COMPONENTS)))
    if len(settings.container_ids) < max(x["diesel"] for x in designs):
        raise ValueError("Missing mappings for grid diesel modules")
    archive = PathArchive(output / "checkpoints" / "paths.jsonl")
    noise_cache = {}
    incumbent, visited, evaluations = None, [], []
    remaining, pruned = [], []
    started = time.perf_counter()
    for index, units in enumerate(designs):
        if max_designs is not None and len(visited) >= max_designs:
            remaining = designs[index:]
            break
        if config["planning"]["investment_bound_pruning"] and incumbent and investment(units) >= incumbent["objective_yuan"]:
            pruned.append(units)
            continue
        for policy in policies:
            if progress:
                progress(dict(stage="training", design=index + 1, total_designs=len(designs), units=units,
                              policy=policy.name, incumbent_yuan=incumbent["objective_yuan"] if incumbent else None))
            evaluated = evaluate_policy(data, weather, units, policy, settings, config["reliability"], archive,
                                        early_screen=config["planning"]["early_risk_screen"], noise_cache=noise_cache)
            record = dict(units=units, policy=policy.name, **{k: v for k, v in evaluated.items() if k != "records"},
                          scenarios_completed=len(evaluated["records"]))
            evaluations.append(record)
            if evaluated["feasible"] and (incumbent is None or evaluated["objective_yuan"] < incumbent["objective_yuan"]):
                incumbent = dict(units=units, capacities=data.capacities(units), policy=asdict(policy),
                                 policy_sha256=policy.fingerprint,
                                 **{k: v for k, v in evaluated.items() if k != "records"})
        visited.append(dict(units=units, scope="finite_policy_library", reason="all_library_policies_resolved",
                            physical_infeasibility_proven=False))
    if incumbent:
        # Completed designs are covered by their exact best feasible library cost;
        # pruned points by investment >= incumbent; every unvisited point remains.
        lower = min([incumbent["objective_yuan"]] + [investment(x) for x in remaining])
        gap = max(0.0, (incumbent["objective_yuan"] - lower) / max(1, abs(incumbent["objective_yuan"])))
        status = "sample_optimal_within_policy_library_gap" if gap == 0 else "sample_feasible_policy_library"
    else:
        lower = min([investment(x) for x in remaining], default=None)
        gap = None
        status = "unresolved" if remaining else "no_feasible_policy_in_declared_grid"
    return dict(status=status, incumbent=incumbent, objective_lower_bound_yuan=lower, relative_gap=gap,
                gap_scope="fixed_grid_fixed_samples_frozen_policy_library", population_certified=False,
                visited=visited, pruned_by_investment_bound=pruned, evaluations=evaluations,
                unresolved_designs=remaining, total_designs=len(designs),
                training_seconds=time.perf_counter() - started,
                bound_model_assumptions=["All operation costs nonnegative", "Investment lower bound only",
                                         "No capacity-monotonicity or legacy availability cut"],
                advanced_recovery_lp_cuts_implemented=False)
