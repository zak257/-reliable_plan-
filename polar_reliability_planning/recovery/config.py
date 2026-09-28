"""Fail-fast schema for recovery-v1; no silent physical parameter defaults."""
from dataclasses import dataclass
import math
from pathlib import Path
import tomllib

from .container_thermal import ThermalParameters
from ..reliability.recovery_policy import RecoveryPolicy


@dataclass(frozen=True)
class RecoverySettings:
    thermal: ThermalParameters
    container_ids: tuple[int, ...]
    crews: int
    prep_hours: int
    personnel_mode: str
    remote_heat: bool
    repair_handover_prepared: bool
    diesel_mttf_hours: float
    start_failure_probability: float
    diesel_repair_mean_hours: float
    repair_distribution: str
    wind_mode: str
    hazard_multiplier: float
    min_output_fraction: float
    min_up_hours: int
    min_down_hours: int
    initial_soc: float
    warmup_hours: int
    fuel_kwh_per_kg: float
    startup_cost_yuan: float
    storage_cost_yuan_per_kwh: float
    failures: dict
    physical_mode: str = "R3"

    def __post_init__(self):
        for name in ("crews", "prep_hours", "min_up_hours", "min_down_hours", "warmup_hours"):
            value = getattr(self, name)
            if isinstance(value, bool) or value != int(value) or value < (1 if name == "crews" else 0):
                raise ValueError(f"{name} must be a valid integer")
        for name in ("diesel_mttf_hours", "diesel_repair_mean_hours", "fuel_kwh_per_kg"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid {name}")
        if self.diesel_repair_mean_hours < 1:
            raise ValueError("Calendar repair mean must be >= 1 hour")
        for name in ("start_failure_probability", "min_output_fraction", "initial_soc"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"Invalid {name}")
        for name in ("hazard_multiplier", "startup_cost_yuan", "storage_cost_yuan_per_kwh"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"Invalid {name}")
        if self.personnel_mode not in ("arrival_only", "all_work_safe"):
            raise ValueError("Invalid personnel_mode")
        if self.wind_mode not in ("protective_shutdown", "operate_with_hazard_multiplier"):
            raise ValueError("Invalid wind_mode")
        if self.physical_mode not in ("R0", "R1", "R2", "R3"):
            raise ValueError("Invalid physical_mode")
        if self.repair_distribution not in ("fixed", "geometric"):
            raise ValueError("Invalid repair distribution")
        if self.repair_distribution == "fixed" and self.diesel_repair_mean_hours != int(self.diesel_repair_mean_hours):
            raise ValueError("Fixed repair duration must be integer")
        if any(isinstance(i, bool) or int(i) != i or i < 0 for i in self.container_ids):
            raise ValueError("container_ids must be nonnegative integers")
        for component in ("wind", "pv", "pcs"):
            law = self.failures[component]
            if not math.isfinite(law["rate_per_hour"]) or law["rate_per_hour"] < 0:
                raise ValueError(f"Invalid {component} failure rate")
            if not math.isfinite(law["repair_mean_hours"]) or law["repair_mean_hours"] < 1:
                raise ValueError(f"Invalid {component} repair mean")


def settings_from_config(config):
    r, d, u = config["recovery"], config["diesel"], config["unit_commitment"]
    return RecoverySettings(
        thermal=ThermalParameters(**config["thermal"]),
        container_ids=tuple(r["container_ids"]), crews=r["crews"], prep_hours=r["prep_hours"],
        personnel_mode=r["personnel_mode"], remote_heat=r["remote_heat"],
        repair_handover_prepared=r["repair_handover_prepared"],
        diesel_mttf_hours=d["mttf_hours"], start_failure_probability=d["start_failure_probability"],
        diesel_repair_mean_hours=d["repair_mean_hours"], repair_distribution=d["repair_distribution"],
        wind_mode=config["weather"]["wind_mode"], hazard_multiplier=config["weather"]["hazard_multiplier"],
        min_output_fraction=u["min_output_fraction"], min_up_hours=u["min_up_hours"],
        min_down_hours=u["min_down_hours"], initial_soc=r["initial_soc"], warmup_hours=r["warmup_hours"],
        fuel_kwh_per_kg=d["fuel_kwh_per_kg"], startup_cost_yuan=u["startup_cost_yuan"],
        storage_cost_yuan_per_kwh=config["cost"]["storage_cost_yuan_per_kwh"],
        failures=config["failures"], physical_mode=r.get("physical_mode", "R3"))


def read_recovery_config(path):
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    if config["schema_version"] != "recovery-v1":
        raise ValueError("Expected schema_version=recovery-v1")
    if config["modules"]["diesel"] != 100:
        raise ValueError("Recovery-v1 research diesel module must be 100 kW")
    if config["objective_basis"] != "expected_causal_operation":
        raise ValueError("Only same-policy actual expected operation cost is supported")
    contract = config["data_contract"]
    if contract["load_basis"] not in ("metered_bus_electricity", "assumed_bus_electricity"):
        raise ValueError("Load must be bus electricity, not unconverted electric+thermal kW")
    if contract["heater_accounting"] not in ("additional_auxiliary_assumption", "verified_not_in_historical_load"):
        raise ValueError("Provide an explicit non-duplicated or assumed additional heater accounting")
    if not config["synthetic"]:
        required = ("thermal_source", "container_mapping_source", "crew_source", "diesel_rating_source",
                    "weather_label_source", "fuel_source", "load_source")
        missing = [k for k in required if not contract.get(k) or contract[k].startswith("assumed")]
        if missing or contract["load_basis"] != "metered_bus_electricity" or contract["heater_accounting"] != "verified_not_in_historical_load":
            raise ValueError(f"Formal run lacks confirmed engineering inputs/accounting: {missing}")
        if config["weather"]["label_mode"] != "observed_csv":
            raise ValueError("Formal run requires sourced joint hourly weather labels")
    if config["diesel"]["rating_basis"] not in ("research_rating", "confirmed_prime", "confirmed_standby"):
        raise ValueError("Declare diesel rating interpretation")
    settings_from_config(config)
    policies = [RecoveryPolicy(**p) for p in config["policies"]]
    if not policies or len({p.name for p in policies}) != len(policies):
        raise ValueError("Nonempty unique frozen policy names required")
    rel = config["reliability"]
    for key in ("samples", "validation_samples"):
        if isinstance(rel[key], bool) or int(rel[key]) != rel[key] or rel[key] < (1 if key == "samples" else 0):
            raise ValueError(f"Invalid {key}")
    if rel["seed"] == rel["validation_seed"]:
        raise ValueError("Training and holdout seeds must be distinct")
    for key in ("seed", "validation_seed"):
        if isinstance(rel[key], bool) or int(rel[key]) != rel[key] or rel[key] < 0:
            raise ValueError(f"Invalid {key}")
    if not 0 <= rel["alpha"] < 1 or any(not math.isfinite(rel[k]) or rel[k] < 0 for k in ("eens_limit_kwh", "cvar_limit_kwh")):
        raise ValueError("Invalid risk thresholds")
    return config
