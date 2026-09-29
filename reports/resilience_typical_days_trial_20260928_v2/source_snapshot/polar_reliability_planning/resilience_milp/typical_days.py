"""Representative controls on a chronological annual clock.

Ordinary days share decisions, not inventories or temperature. The seven input
profiles are replaced by actual medoid days; every original hourly constraint
is then enforced on that reconstructed calendar. This is an approximation and
a policy restriction, neither a relaxation nor an original-year certificate.
"""
from dataclasses import dataclass, replace
from itertools import combinations
import time

import numpy as np
from gurobipy import GRB

from .annual import AnnualResilienceMILP


FEATURES = ('core_kw', 'rigid_kw', 'flex_interruptible_kw', 'flex_shiftable_kw',
            'wind_clean_pu', 'pv_pu', 'ambient_c')


@dataclass
class DayAggregation:
    mapping: np.ndarray
    protected: dict
    representatives: tuple
    feature_scale: np.ndarray

    def check(self, hours):
        if hours % 24 or self.mapping.shape != (hours // 24,):
            raise ValueError('aggregation requires a complete daily calendar')
        if not np.issubdtype(self.mapping.dtype, np.integer):
            raise ValueError('representative day indices must be integers')
        if (self.mapping < 0).any() or (self.mapping >= len(self.mapping)).any():
            raise ValueError('representative day out of range')
        if not np.array_equal(self.mapping[self.mapping], self.mapping):
            raise ValueError('representatives must map to themselves')
        if any(self.mapping[d] != d for d in self.protected):
            raise ValueError('protected days cannot be replaced')

    def reconstruct(self, year):
        self.check(year.hours)
        return replace(year, **{name: getattr(year, name).reshape(-1, 24)[self.mapping].reshape(-1).copy()
                               for name in FEATURES})

    def report(self, year):
        reconstructed = self.reconstruct(year)
        weights = np.bincount(self.mapping, minlength=len(self.mapping))
        errors = {}
        for name in FEATURES:
            a, b = getattr(year, name), getattr(reconstructed, name)
            errors[name] = dict(rmse=float(np.sqrt(np.mean((a-b)**2))),
                original_peak=float(a.max()), reconstructed_peak=float(b.max()),
                annual_sum_relative_error=None if name == 'ambient_c' or a.sum() == 0
                    else float(b.sum()/a.sum()-1))
        return dict(method='monthly actual-day k-medoids, standardized squared hourly distance',
            approximation=True, calendar_days=len(self.mapping),
            ordinary_typical_days=len(self.representatives), protected_days=len(self.protected),
            distinct_dispatch_days=int(np.count_nonzero(weights)), represented_days=int(weights.sum()),
            day_to_representative=self.mapping.tolist(),
            day_weights={str(d): int(w) for d, w in enumerate(weights) if w},
            protected_reasons={str(d): reasons for d, reasons in sorted(self.protected.items())},
            ordinary_representatives=list(self.representatives),
            feature_scale=dict(zip(FEATURES, self.feature_scale.tolist())), input_errors=errors,
            state_policy='all calendar-hour inventories, temperatures and chronological constraints retained; ordinary controls shared',
            objective_policy='calendar occurrences provide day weights; stress costs replace their original hours once')


def cluster_days(year, window_starts=(816, 2496, 5616, 7536), per_month=2, extra_days=()):
    """Protect stress windows/halos, endpoints and capacity-relevant extremes."""
    if year.hours % 24 or not 1 <= per_month <= 3:
        raise ValueError('complete days and 1..3 medoids per month required')
    days = year.hours // 24
    protected = {}
    def protect(a, z, reason):
        for d in range(max(0, a), min(days, z)):
            protected.setdefault(d, []).append(reason)
    protect(0, 3, 'initial startup and preparation')
    protect(days-2, days, 'terminal inventories and outstanding jobs')
    for start in window_starts:
        if start % 24 or start < 0 or start+72 > year.hours:
            raise ValueError('stress window must fit the daily calendar')
        protect(start//24-1, start//24+4, '72h stress window plus 24h boundary halos')
    load = sum(getattr(year, f) for f in FEATURES[:4])
    for values, label, maximum in ((year.core_kw, 'core peak', True),
                                  (load, 'total load peak', True),
                                  (year.ambient_c, 'minimum temperature', False)):
        day = int(np.argmax(values) if maximum else np.argmin(values)) // 24
        protect(day-1, day+2, label+' and adjacent days')
    if days >= 3:
        wind = year.wind_clean_pu / max(float(year.wind_clean_pu.mean()), 1e-12)
        pv = year.pv_pu / max(float(year.pv_pu.mean()), 1e-12)
        renewable = (wind+pv).reshape(days, 24).mean(axis=1)
        deficit = load.reshape(days, 24).mean(axis=1)/max(float(load.mean()), 1e-12)-renewable/2
        low = int(np.argmin(np.convolve(renewable, np.ones(3), 'valid')))
        high = int(np.argmax(np.convolve(deficit, np.ones(3), 'valid')))
        protect(low-1, low+4, 'lowest renewable 72h plus boundary halos')
        protect(high-1, high+4, 'high load low renewable 72h plus boundary halos')
    for day in extra_days:
        if not 0 <= day < days: raise ValueError('extra retained day outside calendar')
        protect(day-1, day+2, 'adaptive enrichment and adjacent days')
    ordinary = np.array([d not in protected for d in range(days)])
    raw = np.stack([getattr(year, f).reshape(days, 24) for f in FEATURES], axis=2)
    sample = raw[ordinary] if ordinary.any() else raw
    scale = sample.reshape(-1, len(FEATURES)).std(axis=0)
    scale[scale < 1e-12] = 1.
    x = (raw/scale).reshape(days, -1)
    months = np.array([str(t)[:7] for t in year.timestamps[::24]])
    mapping = np.arange(days)
    representatives = []
    for month in sorted(set(months)):
        ids = np.flatnonzero((months == month) & ordinary)
        if not len(ids): continue
        distances = ((x[ids, None, :]-x[None, ids, :])**2).mean(axis=2)
        candidates = np.array(list(combinations(range(len(ids)), min(per_month, len(ids)))))
        costs = distances[:, candidates].min(axis=2).sum(axis=0)
        medoids = candidates[int(np.argmin(costs))]
        nearest = np.argmin(distances[:, medoids], axis=1)
        mapping[ids] = ids[medoids[nearest]]
        # Identical medoids must remain fixed points even under tied distances.
        mapping[ids[medoids]] = ids[medoids]
        representatives.extend(ids[medoids].tolist())
    result = DayAggregation(mapping, protected, tuple(representatives), scale)
    result.check(year.hours)
    return result


def remove_exact_duplicate_constraints(model):
    """Remove only byte-identical linear rows; all chronological rows survive.

    No rounding or tolerance is used. General constraints are left to Gurobi.
    This is structural deduplication of the already constructed approximation.
    """
    started = time.monotonic()
    model.update()
    matrix = model.getA().tocsr()
    matrix.sum_duplicates(); matrix.sort_indices(); matrix.eliminate_zeros()
    constraints = model.getConstrs()
    senses = model.getAttr('Sense', constraints)
    rhs = model.getAttr('RHS', constraints)
    seen = set(); redundant = []
    for row, constraint in enumerate(constraints):
        a, z = matrix.indptr[row:row+2]
        key = (senses[row], rhs[row], matrix.indices[a:z].tobytes(), matrix.data[a:z].tobytes())
        if key in seen: redundant.append(constraint)
        else: seen.add(key)
    model.remove(redundant); model.update()
    return dict(removed_linear_constraints=len(redundant), remaining_linear_constraints=model.NumConstrs,
                seconds=time.monotonic()-started)


class TypicalDayResilienceMILP(AnnualResilienceMILP):
    """Original physics on a reconstructed year, with shared ordinary controls."""
    def __init__(self, base, windows, cfg, aggregation, original_year, **kwargs):
        aggregation.check(base.year.hours)
        self.aggregation = aggregation
        self.original_year = original_year
        self.shared_control_entries = 0
        for window in windows:
            for d in range(window['start']//24, window['stop']//24):
                if aggregation.mapping[d] != d:
                    raise ValueError('stress-window days must be retained individually')
        expected = aggregation.reconstruct(original_year)
        for name in FEATURES:
            if not np.array_equal(getattr(expected, name), getattr(base.year, name)):
                raise ValueError('base data do not match the aggregation map')
        kwargs['compact'] = True
        super().__init__(base, windows, cfg, **kwargs)
        # Derive sizing constants from the original calendar even if a future
        # clustering recipe does not select every relevant peak explicitly.
        peak = float(original_year.core_kw.max())
        self.required_ups_kwh = peak*cfg.ups_bridge_hours/((cfg.ups_standby_soc-cfg.ups_min_soc)*cfg.ups_efficiency)
        self.required_ups_kw = peak*cfg.ups_power_margin
        self.model.addConstr(self.cap['ups_kwh'] >= self.required_ups_kwh, name='original_year_ups_energy')
        self.model.addConstr(self.cap['ups_kw'] >= self.required_ups_kw, name='original_year_ups_power')
        self.model.update()

    def _dispatch_vars(self, prefix, name, keys, **attributes):
        if prefix != 's0_' or name in ('battery_energy', 'ups_energy', 'temperature'):
            return super()._dispatch_vars(prefix, name, keys, **attributes)
        def hour(t):
            return int(self.aggregation.mapping[t//24])*24+t%24
        def canonical(key):
            if name == 'shift_service':
                a, t = key
                return hour(a), hour(a)+(t-a)
            if isinstance(key, tuple): return key[0], hour(key[1])
            return hour(key)
        aliases = {key: canonical(key) for key in keys}
        unique = list(dict.fromkeys(aliases.values()))
        variables = super()._dispatch_vars(prefix, name, unique, **attributes)
        self.shared_control_entries += len(keys)-len(unique)
        return {key: variables[value] for key, value in aliases.items()}

    def result(self):
        result = super().result()
        result.update(model_scope='approximate_typical_day_controls_with_chronological_states',
            full_year_original_data_validated=False, original_year_optimality_certified=False,
            economic_domain_certified=False,
            shared_control_entries=self.shared_control_entries,
            distinct_dispatch_days=len(np.unique(self.aggregation.mapping)),
            ordinary_typical_days=len(self.aggregation.representatives),
            protected_days=len(self.aggregation.protected),
            approximation='medoid input profiles and shared ordinary-day controls; all chronological constraints retained')
        if result.get('selected'):
            result['status'] = ('approximate_model_optimal_within_gap' if self.model.Status == GRB.OPTIMAL
                                else 'approximate_model_feasible_incumbent')
        return result
