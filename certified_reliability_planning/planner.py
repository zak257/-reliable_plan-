"""Finite-grid certificate search with cost priorities and a fair sweep.

All cost lower bounds remain in the global minimum until a point has a
reliability failure certificate or a proven infeasible nominal dispatch.
Total cost need not be monotone. Resource limits produce an unresolved result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from .risk import risk_bounds


@dataclass(frozen=True)
class PlannerOptions:
    delta: float = 0.05
    epsilon_cost: float = 1.0
    relative_gap: float | None = None
    initial_samples: int = 16
    max_samples: int = 1024
    max_stages: int = 10
    max_oracle_calls: int = 2000
    max_economic_calls: int = 200
    max_seconds: float = 600.0  # Zero disables the total wall-clock budget.
    base_call_seconds: float = 0.1
    max_call_seconds: float = 30.0
    fairness_interval: int = 4

    def __post_init__(self):
        if not math.isfinite(self.delta) or not 0 < self.delta < 1:
            raise ValueError("delta must be in (0,1)")
        if self.relative_gap is not None and (isinstance(self.relative_gap, bool)
                or not math.isfinite(self.relative_gap) or not 0 < self.relative_gap < 1):
            raise ValueError("relative_gap must be in (0,1)")
        if (isinstance(self.max_seconds, bool) or not math.isfinite(self.max_seconds)
                or self.max_seconds < 0):
            raise ValueError("max_seconds must be finite and nonnegative; 0 means no total time limit")
        for name in ("epsilon_cost", "base_call_seconds", "max_call_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("initial_samples", "max_samples", "max_stages", "max_oracle_calls",
                     "max_economic_calls", "fairness_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.initial_samples > self.max_samples:
            raise ValueError("initial_samples cannot exceed max_samples")
        if self.base_call_seconds > self.max_call_seconds:
            raise ValueError("base_call_seconds cannot exceed max_call_seconds")

    def cost_tolerance(self, upper_cost: float) -> float:
        """Relative mode replaces the absolute criterion; it is never an OR."""
        return self.epsilon_cost if self.relative_gap is None else self.relative_gap * abs(upper_cost)


@dataclass
class PointRecord:
    lower: np.ndarray
    upper: np.ndarray
    risk: object | None = None


class CertifiedPlanner:
    def __init__(self, candidates, cost_lower_bounds, stream, operation, economic,
                 options: PlannerOptions, eens_limit: float, cvar_limit: float,
                 alpha: float, on_event=None, region_search=None):
        raw = np.asarray(candidates)
        if (raw.ndim != 2 or min(raw.shape) < 1 or not np.isfinite(raw).all()
                or np.any(raw < 0) or np.any(raw != np.floor(raw))):
            raise ValueError("candidates must be a nonempty matrix of nonnegative integers")
        self.points = raw.astype(np.int64)
        self.keys = [tuple(map(int, p)) for p in self.points]
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("duplicate capacity candidates")
        self.index = {p: i for i, p in enumerate(self.keys)}
        self.cost_lower = np.asarray(cost_lower_bounds, dtype=float).copy()
        if (self.cost_lower.shape != (len(self.points),) or not np.isfinite(self.cost_lower).all()
                or np.any(self.cost_lower < 0)):
            raise ValueError("one finite nonnegative initial cost lower bound is required per point")
        for limit in (eens_limit, cvar_limit):
            if not math.isfinite(limit) or limit < 0:
                raise ValueError("risk limits must be finite and nonnegative")
        if not math.isfinite(alpha) or not 0 <= alpha < 1:
            raise ValueError("alpha must be in [0,1)")
        self.stream, self.operation, self.economic = stream, operation, economic
        self.bound = float(stream.loss_bound)
        if not math.isfinite(self.bound) or self.bound < 0:
            raise ValueError("stream must provide a finite deterministic loss bound")
        self.options = options
        self.eens_limit, self.cvar_limit, self.alpha = eens_limit, cvar_limit, alpha
        self.on_event = on_event
        self.region_search = region_search
        self.region_lower = 0.0
        self.region_infeasible = False
        self.region_dirty = region_search is not None
        self.region_suggestion = None
        self.cost_upper = np.full(len(self.points), np.inf)
        self.labels = np.zeros(len(self.points), dtype=np.int8)
        self.records: dict[tuple, PointRecord] = {}
        self.witnesses: list[dict] = []
        self.economic_dispatches: dict[tuple, dict] = {}
        self.counters = {"annual_samples": 0, "distinct_operation_pairs": 0, "operation_calls": 0,
                         "economic_calls": 0, "operation_work": 0.0, "economic_work": 0.0,
                         "operation_seconds": 0.0, "economic_seconds": 0.0,
                         "generation_seconds": 0.0, "fair_selections": 0, "region_calls": 0}
        self.queried_pairs = set()
        self.events = 0
        self.stages_completed = 0
        self.cursor = 0
        self.started = None
        self.stop_reason = None
        self._has_run = False

    def _emit(self, action, **values):
        self.events += 1
        if self.on_event:
            self.on_event({"event": self.events, "action": action,
                           "elapsed_seconds": time.perf_counter() - self.started, **values})

    def _time_left(self):
        if self.options.max_seconds == 0:
            return math.inf
        return self.options.max_seconds - (time.perf_counter() - self.started)

    def _remaining(self, kind):
        if self._time_left() <= 0:
            self.stop_reason = "wall_time_budget"
            return False
        limit = self.options.max_oracle_calls if kind == "operation" else self.options.max_economic_calls
        if self.counters[kind + "_calls"] >= limit:
            self.stop_reason = kind + "_call_budget"
            return False
        return True

    def _incumbent(self):
        indices = np.flatnonzero((self.labels == 1) & np.isfinite(self.cost_upper))
        return int(indices[np.argmin(self.cost_upper[indices])]) if len(indices) else None

    def _global_lower(self):
        live = (self.labels != -1) & (self.labels != 2)
        if self.region_infeasible or not np.any(live):
            return math.inf
        return max(self.region_lower, float(np.min(self.cost_lower[live])))

    def _certificate_status(self):
        if self.region_infeasible:
            if self._incumbent() is not None:
                raise ValueError("Infeasible economic region contradicts a certified incumbent")
            return "certified_infeasible"
        if np.all((self.labels == -1) | (self.labels == 2)):
            return "certified_infeasible"
        best = self._incumbent()
        if best is not None:
            lo, hi = self._global_lower(), float(self.cost_upper[best])
            if lo > hi:
                raise ValueError("Economic bounds contradict the incumbent")
            if hi - lo <= self.options.cost_tolerance(hi):
                return "certified_optimal"
        return None

    def _competitive(self):
        live = (self.labels != -1) & (self.labels != 2)
        best = self._incumbent()
        if best is not None:
            # Positive slack supports finite termination as economic bounds converge.
            live &= self.cost_lower < self.cost_upper[best] - self.options.cost_tolerance(self.cost_upper[best]) / 2
            live[best] = True
        return live

    def _record(self, point):
        m = self.stream.samples
        record = self.records.get(point)
        if record is None:
            record = PointRecord(np.zeros(m), np.full(m, self.bound))
            self.records[point] = record
        elif len(record.lower) < m:
            count = m - len(record.lower)
            record.lower = np.concatenate((record.lower, np.zeros(count)))
            record.upper = np.concatenate((record.upper, np.full(count, self.bound)))
        # Every donor refers to the identical sample index and module prefix.
        for other, donor in self.records.items():
            if other == point:
                continue
            size = min(m, len(donor.lower))
            if all(a <= b for a, b in zip(point, other)):
                record.lower[:size] = np.maximum(record.lower[:size], donor.lower[:size])
            if all(a >= b for a, b in zip(point, other)):
                record.upper[:size] = np.minimum(record.upper[:size], donor.upper[:size])
        if np.any(record.lower > record.upper):
            raise ValueError("Contradictory path bounds; check capacity monotonicity and Oracle validity")
        return record

    def _classify(self, index, record):
        record.risk = risk_bounds(record.lower, record.upper, self.bound, self.alpha,
                                  self.options.delta, len(self.points))
        r = record.risk
        if r.eens_upper <= self.eens_limit and r.cvar_upper <= self.cvar_limit:
            label, covered = 1, np.all(self.points >= self.points[index], axis=1)
        elif r.eens_lower > self.eens_limit or r.cvar_lower > self.cvar_limit:
            label, covered = -1, np.all(self.points <= self.points[index], axis=1)
        else:
            return False
        if np.any(self.labels[covered] == -label):
            raise ValueError("Conflicting statistical labels; no certificate may be returned")
        covered &= self.labels != 2
        changed = int(np.count_nonzero(covered & (self.labels == 0)))
        self.labels[covered] = label
        witness = {"point": list(self.keys[index]), "label": "feasible" if label == 1 else "infeasible",
                   "sample_count": self.stream.samples, "risk": asdict(r), "newly_classified": changed}
        self.witnesses.append(witness)
        self._emit("risk_certificate", **witness)
        if label == -1 and self.region_search is not None:
            self.region_search.add_failure(self.keys[index])
            self.region_dirty = True
        return True

    def _solve_region(self, stage):
        if not self._remaining("economic"):
            return
        budget = min(self.options.max_call_seconds,
                     self.options.base_call_seconds * 2.0 ** min(stage, 60), max(1e-6, self._time_left()))
        # In relative mode the economic solver uses its configured relative
        # MIP gap. Disable absolute-gap early stopping so 1% is respected.
        absolute_gap = (0.0 if self.options.relative_gap is not None else
                        self.options.epsilon_cost * 2.0 ** (-stage - 2))
        result = self.region_search.solve(budget, absolute_gap)
        self._count("economic", result)
        self.counters["region_calls"] += 1
        if math.isnan(result.lower) or result.lower < 0:
            raise ValueError("Invalid regional economic lower bound")
        self.region_lower = max(self.region_lower, float(result.lower))
        self.region_infeasible = bool(result.infeasible)
        if not self.region_infeasible and not math.isfinite(self.region_lower):
            raise ValueError("Infinite regional lower bound without infeasibility proof")
        self.region_suggestion = None
        if result.point is not None:
            point = tuple(result.point)
            if point not in self.index:
                raise ValueError("Economic search returned a point outside the declared capacity grid")
            index = self.index[point]
            if self.labels[index] in (-1, 2):
                raise ValueError("Economic search returned an excluded point")
            if result.upper is not None:
                if not math.isfinite(result.upper) or result.upper < self.region_lower:
                    raise ValueError("Invalid regional incumbent cost upper bound")
                if result.upper < self.cost_lower[index]:
                    raise ValueError("Regional incumbent contradicts point cost lower bound")
                if result.upper < self.cost_upper[index]:
                    self.cost_upper[index] = result.upper
                    if result.dispatch is not None:
                        self.economic_dispatches[point] = result.dispatch
            self.cost_lower[index] = max(self.cost_lower[index], self.region_lower)
            self.region_suggestion = index
        self.region_dirty = False
        self._emit("economic_region", lower=float(result.lower), point=result.point,
                   upper=result.upper, status=result.status, infeasible=result.infeasible)

    @staticmethod
    def _validate_interval(result, allow_infinite=False):
        lo, hi = float(result.lower), float(result.upper)
        if math.isnan(lo) or math.isnan(hi) or lo < 0 or lo > hi:
            raise ValueError("Oracle returned invalid interval")
        if not allow_infinite and not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError("Operation bounds must be finite")
        for name in ("work", "runtime_seconds"):
            value = float(getattr(result, name, 0.0))
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid Oracle work counter")
        return lo, hi

    def _count(self, kind, result):
        self.counters[kind + "_calls"] += 1
        self.counters[kind + "_work"] += float(getattr(result, "work", 0.0))
        self.counters[kind + "_seconds"] += float(getattr(result, "runtime_seconds", 0.0))

    def _visit(self, index, stage):
        point = self.keys[index]
        budget = min(self.options.max_call_seconds, self.options.base_call_seconds * 2.0 ** min(stage, 60))
        economic_gap = (0.0 if self.options.relative_gap is not None else
                        self.options.epsilon_cost * 2.0 ** (-stage - 2))
        if self.cost_upper[index] - self.cost_lower[index] > economic_gap:
            if not self._remaining("economic"):
                return
            result = self.economic(point, min(budget, max(1e-6, self._time_left())), economic_gap)
            lo, hi = self._validate_interval(result, allow_infinite=True)
            self._count("economic", result)
            self.cost_lower[index] = max(self.cost_lower[index], lo)
            if hi < self.cost_upper[index]:
                self.cost_upper[index] = hi
                if getattr(result, "dispatch", None) is not None:
                    self.economic_dispatches[point] = result.dispatch
            if self.cost_lower[index] > self.cost_upper[index]:
                raise ValueError("Economic Oracle intervals are inconsistent")
            if lo == math.inf:
                self.labels[index] = 2
            self._emit("economic", point=list(point), lower=lo, upper=hi,
                       status=getattr(result, "status", "unknown"))
        if self.labels[index] != 0 or not self._competitive()[index]:
            return
        record = self._record(point)
        if self._classify(index, record):
            return
        target_width = self.bound * 2.0 ** (-stage - 1)
        # Worst intervals first; each pair is visited at most once per stage.
        order = np.argsort(-(record.upper - record.lower), kind="stable")
        since_check = 0
        total_width = float(np.sum(record.upper - record.lower))
        for s in order:
            width = record.upper[s] - record.lower[s]
            if width <= 0 or total_width / len(record.lower) <= target_width:
                break
            if not self._remaining("operation"):
                return
            result = self.operation(point, int(s), min(budget, max(1e-6, self._time_left())),
                                    target_width / 2)
            lo, hi = self._validate_interval(result)
            if hi > self.bound or lo > self.bound:
                raise ValueError("Operation Oracle exceeds declared deterministic loss bound")
            self._count("operation", result)
            self.queried_pairs.add((point, int(s)))
            self.counters["distinct_operation_pairs"] = len(self.queried_pairs)
            record.lower[s] = max(record.lower[s], lo)
            record.upper[s] = min(record.upper[s], hi)
            if record.lower[s] > record.upper[s]:
                raise ValueError("Refinement contradicts previous valid path bounds")
            total_width -= width - (record.upper[s] - record.lower[s])
            self._emit("operation", point=list(point), sample=int(s), lower=float(record.lower[s]),
                       upper=float(record.upper[s]), status=getattr(result, "status", "unknown"))
            since_check += 1
            if since_check >= 8:
                if self._classify(index, record):
                    return
                since_check = 0
        self._classify(index, record)

    def run(self):
        if self._has_run:
            raise RuntimeError("Create a new planner for each run")
        self._has_run = True
        self.started = time.perf_counter()
        status = None
        for stage in range(self.options.max_stages):
            if self._time_left() <= 0:
                self.stop_reason = "wall_time_budget"
                break
            m = self.stream.samples
            if m == 0:
                target = self.options.initial_samples
            else:
                competitive = self._competitive()
                unresolved = [r for p, r in self.records.items()
                              if self.labels[self.index[p]] == 0 and competitive[self.index[p]]
                              and r.risk is not None]
                oracle_width = max((r.risk.mean_oracle_width for r in unresolved), default=0.0)
                statistical_width = max((2 * self.bound * r.risk.radius for r in unresolved), default=self.bound)
                target = min(self.options.max_samples, 2 * m) if oracle_width <= statistical_width else m
            before = time.perf_counter()
            self.stream.extend(target)
            if self.stream.samples != target:
                raise ValueError("Sampling Oracle did not return the requested prefix")
            self.counters["generation_seconds"] += time.perf_counter() - before
            self.counters["annual_samples"] = target
            self._emit("stage", stage=stage + 1, sample_count=target,
                       interval_target=self.bound * 2.0 ** (-stage - 1),
                       lower_bound_cost=self._global_lower())
            if self.region_search is not None:
                self.region_dirty = True
            visited = np.zeros(len(self.points), dtype=bool)
            selection = 0
            while True:
                status = self._certificate_status()
                if status or self.stop_reason:
                    break
                if self.region_dirty:
                    self._solve_region(stage)
                    status = self._certificate_status()
                    if status or self.stop_reason:
                        break
                # Regional optimization targets the cheapest remaining design;
                # one finite cyclic visit per stage protects neglected designs.
                if self.region_search is not None and selection >= self.options.fairness_interval:
                    break
                eligible = np.flatnonzero(self._competitive() & ~visited)
                if not len(eligible):
                    break
                if selection % self.options.fairness_interval == self.options.fairness_interval - 1:
                    after = eligible[eligible >= self.cursor]
                    index = int(after[0] if len(after) else eligible[0])
                    self.cursor = (index + 1) % len(self.points)
                    self.counters["fair_selections"] += 1
                elif self.region_suggestion is not None and self.region_suggestion in eligible:
                    index = self.region_suggestion
                else:
                    index = int(eligible[np.argmin(self.cost_lower[eligible])])
                visited[index] = True
                selection += 1
                self._visit(index, stage)
            self.stages_completed = stage + 1
            if status or self.stop_reason:
                break
        status = status or self._certificate_status() or "budget_exhausted"
        if status == "budget_exhausted" and self.stop_reason is None:
            self.stop_reason = "stage_or_sample_budget"
        return self.result(status)

    def result(self, status):
        best = self._incumbent()
        lower = self._global_lower()
        upper = float(self.cost_upper[best]) if best is not None else None
        criterion = {"kind": "absolute_cost" if self.options.relative_gap is None else "relative_gap",
                     "epsilon_cost": self.options.epsilon_cost if self.options.relative_gap is None else None,
                     "relative_gap": self.options.relative_gap,
                     "relative_gap_denominator": "abs(feasible_cost_upper_bound)"}
        nominal = np.flatnonzero(np.isfinite(self.cost_upper) & (self.labels != -1) & (self.labels != 2))
        candidate = None
        if len(nominal):
            index = int(nominal[np.argmin(self.cost_upper[nominal])])
            point = self.keys[index]
            candidate_upper = float(self.cost_upper[index])
            record = self._record(point)
            current_risk = risk_bounds(record.lower, record.upper, self.bound, self.alpha,
                                       self.options.delta, len(self.points))
            candidate = {"point": list(point), "cost_upper": candidate_upper,
                         "global_cost_lower": lower, "relative_economic_gap":
                         (candidate_upper - lower) / abs(candidate_upper) if candidate_upper else 0.0,
                         "reliability_certified": bool(self.labels[index] == 1),
                         "risk_bounds": asdict(current_risk)}
        certificate = None
        if status.startswith("certified_"):
            certificate = {"type": status, "delta": self.options.delta,
                           "epsilon_cost": criterion["epsilon_cost"], "optimality_criterion": criterion,
                           "cardinality_bound": len(self.points), "loss_bound": self.bound,
                           "simultaneous_event": "DKW_union_over_all_capacities_and_all_prefix_lengths",
                           "global_lower_bound_includes_all_unexcluded_designs": True,
                           "conditional_on_valid_oracles_and_iid_sampling": True}
        return {"status": status, "stop_reason": self.stop_reason,
                "incumbent": list(self.keys[best]) if best is not None else None,
                "incumbent_reliability_certified": best is not None,
                "lower_bound_cost": lower if math.isfinite(lower) else None,
                "upper_bound_cost": upper,
                "absolute_gap_cost": upper - lower if upper is not None else None,
                "relative_gap_cost": ((upper - lower) / abs(upper) if upper else 0.0) if upper is not None else None,
                "optimality_criterion": criterion, "best_nominal_candidate": candidate,
                "certificate": certificate, "options": asdict(self.options),
                "limits": {"eens": self.eens_limit, "cvar": self.cvar_limit, "alpha": self.alpha},
                "loss_bound": self.bound, "cardinality": len(self.points),
                "labels": {"unknown": int(np.sum(self.labels == 0)),
                           "risk_feasible": int(np.sum(self.labels == 1)),
                           "risk_infeasible": int(np.sum(self.labels == -1)),
                           "nominal_infeasible": int(np.sum(self.labels == 2))},
                "risk_witnesses": self.witnesses, "counters": dict(self.counters),
                "regional_lower_bound": self.region_lower if math.isfinite(self.region_lower) else None,
                "regional_infeasibility_proven": self.region_infeasible,
                "stages_completed": self.stages_completed,
                "elapsed_seconds": time.perf_counter() - self.started}

    def save_state(self, path):
        """Evidence archive, not an automatic resume interface."""
        arrays = {"candidates": self.points, "cost_lower": self.cost_lower,
                  "cost_upper": self.cost_upper, "labels": self.labels}
        for point, record in self.records.items():
            index = self.index[point]
            arrays[f"loss_lower_{index}"] = record.lower
            arrays[f"loss_upper_{index}"] = record.upper
        np.savez_compressed(path, **arrays)
