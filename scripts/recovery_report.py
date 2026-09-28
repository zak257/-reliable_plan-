"""Rebuild plots and compact comparisons from completed, audited runs."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def annotate_recovery_windows(directory, output_path):
    """Retrospective window quantities; overlapping windows must not be summed."""
    hourly = pd.read_csv(directory / "hourly_dispatch.csv")
    try:
        delays = pd.read_csv(directory / "delay_decomposition.csv")
    except pd.errors.EmptyDataError:
        return
    for column, source in (("battery_bridge_kwh", "discharge_kw"), ("user_shed_kwh", "shed_kw"),
                           ("heater_supply_kwh", "heater_kw")):
        delays[column] = [float(hourly[(hourly.hour >= row.request_hour) & (hourly.hour < row.end_hour)][source].sum())
                          for row in delays.itertuples()]
    delays["quantity_scope"] = "whole-system flow during this request window; overlapping windows are not additive"
    delays.to_csv(output_path, index=False)


def dispatch_plot(directory, filename, start=None, end=None):
    df = pd.read_csv(directory / "hourly_dispatch.csv")
    df = df[df.hour >= 0]
    if start is not None:
        df = df[df.hour >= start]
    if end is not None:
        df = df[df.hour < end]
    h = df.hour.to_numpy()
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    axes[0].step(h, df.load_kw, where="post", color="black", label="User load")
    axes[0].step(h, df.diesel_kw, where="post", label="Diesel")
    axes[0].step(h, df.renewable_used_kw, where="post", label="Renewable used")
    axes[0].step(h, df.discharge_kw - df.charge_kw, where="post", label="Battery net discharge")
    axes[0].fill_between(h, df.shed_kw, step="post", color="#c94235", alpha=.45, label="Unserved load")
    axes[0].set_ylabel("Power (kW)")
    axes[0].legend(ncol=3, fontsize=9)
    axes[1].step(h, df.energy_start_kwh, where="post", color="#177e89", label="Actual battery inventory")
    axes[1].set_ylabel("Stored energy (kWh)")
    axes[1].legend(loc="upper right", fontsize=9)
    temperatures = np.array([json.loads(v) for v in df.temperatures_start])
    for i in range(temperatures.shape[1]):
        axes[2].plot(h, temperatures[:, i], label=f"Container {i}", linewidth=1.2)
    axes[2].axhline(5, color="black", linestyle="--", alpha=.6, label="Ready threshold (assumed)")
    axes[2].set_ylabel("Temperature (C)")
    axes[2].legend(ncol=3, fontsize=9)
    online = [sum(s["running"] for s in json.loads(v)) for v in df.diesel_states]
    axes[3].step(h, online, where="post", label="Online diesel units")
    axes[3].step(h, df.crews_onsite, where="post", label="Startup crews onsite")
    upper = max(online, default=1) + .5
    axes[3].fill_between(h, 0, upper, where=~df.access_safe.to_numpy(dtype=bool),
                         step="post", alpha=.16, color="#c94235", label="Access blocked")
    axes[3].set_ylabel("Units / crews")
    axes[3].set_xlabel("Measured hour (continuous state)")
    axes[3].legend(ncol=3, fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("Recovery-v1: hourly physical execution — engineering assumptions apply", fontsize=14)
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    reports = args.root / "reports"
    output = reports / "recovery_comparison_v1"
    output.mkdir(exist_ok=True)
    synthetic = reports / "recovery_synthetic_72h_v3"
    dispatch_plot(synthetic / "physics_72h", output / "synthetic_72h_dispatch.png")
    annotate_recovery_windows(synthetic / "physics_72h", output / "synthetic_recovery_windows.csv")
    comparison = []
    for mode in ("protective", "operating"):
        directory = reports / f"zhongshan_recovery_8760_{mode}_16_128_v1"
        result = json.loads((directory / "summary.json").read_text())
        inc = result["incumbent"]
        comparison.append(dict(wind_rule=mode, **inc["capacities"], policy=inc["policy"]["name"],
                               training_objective_yuan=inc["objective_yuan"],
                               training_eens_kwh=inc["risk"]["eens_kwh"],
                               training_cvar_kwh=inc["risk"]["cvar_kwh"],
                               holdout_eens_kwh=result["validation"]["risk"]["eens_kwh"],
                               holdout_cvar_kwh=result["validation"]["risk"]["cvar_kwh"],
                               holdout_status=result["validation_status"],
                               elapsed_seconds=result["elapsed_seconds"], audit_passed=result["audit_passed"]))
        hourly = pd.read_csv(directory / "hourly_dispatch.csv")
        shed = hourly[(hourly.hour >= 0) & (hourly.shed_kw > 1e-7)]
        events = pd.read_csv(directory / "event_log.csv")
        faults = events[(events.hour >= 0) & (events.kind == "diesel_run_failure")]
        event_hour = int(shed.hour.iloc[0] if len(shed) else faults.hour.iloc[0] if len(faults) else 12)
        dispatch_plot(directory, output / f"{mode}_event_dispatch.png", max(0, event_hour - 12), event_hour + 48)
        annotate_recovery_windows(directory, output / f"{mode}_recovery_windows.csv")
        losses = pd.read_csv(directory / "scenario_losses.csv")
        convergence = []
        # Descriptive fixed-design prefixes only; no sequential confidence claim.
        from polar_reliability_planning.reliability.risk_bounds import risk_summary
        for role, sizes in (("training", [4, 8, 16]), ("holdout", [16, 32, 64, 128])):
            q = losses[losses.role == role].sort_values("scenario").loss_kwh.to_numpy()
            for n in sizes:
                if n <= len(q):
                    convergence.append(dict(role=role, prefix_samples=n, **risk_summary(q[:n], .95)))
        pd.DataFrame(convergence).to_csv(output / f"{mode}_risk_prefixes.csv", index=False)
    pd.DataFrame(comparison).to_csv(output / "wind_rule_comparison.csv", index=False)
    (output / "wind_rule_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    expanded = reports / "zhongshan_recovery_8760_protective_128_256_v1" / "full_grid"
    if (expanded / "summary.json").exists():
        final = json.loads((expanded / "summary.json").read_text())
        hourly = pd.read_csv(expanded / "hourly_dispatch.csv")
        shed = hourly[(hourly.hour >= 0) & (hourly.shed_kw > 1e-7)]
        event_hour = int(shed.hour.iloc[0]) if len(shed) else 100
        dispatch_plot(expanded, output / "expanded_event_dispatch.png", max(0, event_hour - 12), event_hour + 48)
        annotate_recovery_windows(expanded, output / "expanded_recovery_windows.csv")
        inc = final["incumbent"]
        comparison.append(dict(wind_rule="protective_128_training_256_holdout", **inc["capacities"],
                               policy=inc["policy"]["name"], training_objective_yuan=inc["objective_yuan"],
                               training_eens_kwh=inc["risk"]["eens_kwh"], training_cvar_kwh=inc["risk"]["cvar_kwh"],
                               holdout_eens_kwh=final["validation"]["risk"]["eens_kwh"],
                               holdout_cvar_kwh=final["validation"]["risk"]["cvar_kwh"],
                               holdout_status=final["validation_status"], audit_passed=final["audit_passed"]))
        pd.DataFrame(comparison).to_csv(output / "sample_size_comparison.csv", index=False)
        (output / "sample_size_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
