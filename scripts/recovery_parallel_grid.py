"""Partition a finite grid, merge verified exact path caches, then freeze/validate.

Each subprocess runs the unmodified serial planner. Grid partitions are disjoint
wind/PV slices. The final original full-grid planner checks every pair again from
the combined exact cache and computes the global library bound. No worker sees
holdout scenarios. This is engineering parallelism, not a new mathematical cut.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

from polar_reliability_planning.recovery.config import read_recovery_config
from polar_reliability_planning.recovery_cli import ROOT, fingerprint, load_inputs, write_json


def toml_text(value):
    lines = []
    def scalar(x):
        return json.dumps(x, ensure_ascii=False)
    def table(obj, prefix="", header=None):
        if header:
            lines.extend(["", header])
        for key, item in obj.items():
            if not isinstance(item, dict) and not (isinstance(item, list) and item and isinstance(item[0], dict)):
                lines.append(f"{key} = {scalar(item)}")
        for key, item in obj.items():
            name = prefix + "." + key if prefix else key
            if isinstance(item, dict):
                table(item, name, f"[{name}]")
            elif isinstance(item, list) and item and isinstance(item[0], dict):
                for child in item:
                    table(child, name, f"[[{name}]]")
    table(value)
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/zhongshan_recovery_stress.toml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--validation-seed", type=int, default=2026091703)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = read_recovery_config(args.config)
    base["reliability"].update(samples=args.samples, validation_samples=args.validation_samples,
                               validation_seed=args.validation_seed)
    full_config_path = output / "full_config.toml"
    full_config_path.write_text(toml_text(base))
    base = read_recovery_config(full_config_path)
    jobs = []
    for i, (wind, pv) in enumerate(itertools.product(base["grid"]["wind"], base["grid"]["pv"])):
        part = copy.deepcopy(base)
        part["grid"]["wind"], part["grid"]["pv"] = [wind], [pv]
        part["reliability"]["validation_samples"] = 0
        path = output / f"partition_{i}.toml"
        path.write_text(toml_text(part))
        jobs.append((i, path, output / f"partition_{i}"))

    def worker(job):
        i, config_path, directory = job
        with (output / f"partition_{i}.log").open("w") as stream:
            result = subprocess.run([sys.executable, "-m", "polar_reliability_planning.recovery_cli", "plan",
                                     "--config", str(config_path), "--output", str(directory)],
                                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode not in (0, 2):
            raise RuntimeError(f"Partition {i} failed: see {output / f'partition_{i}.log'}")
        print(json.dumps(dict(partition=i, status="complete", elapsed_seconds=time.perf_counter() - started)), flush=True)
        return directory

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        completed = list(pool.map(worker, jobs))
    training_seconds = time.perf_counter() - started
    data, weather, _ = load_inputs(base)
    digest, inputs = fingerprint(base, data, weather)
    final = output / "full_grid"
    (final / "checkpoints").mkdir(parents=True)
    seen = set()
    # Reject mixing any physical, stochastic, cost or policy changes. Only grid
    # wind/PV partitioning and holdout count may differ from the full config.
    with (final / "checkpoints" / "paths.jsonl").open("w") as destination:
        for part_dir in completed:
            manifest = json.loads((part_dir / "run_manifest.json").read_text())
            normalized = copy.deepcopy(manifest["config"])
            normalized["grid"] = base["grid"]
            normalized["reliability"]["validation_samples"] = args.validation_samples
            if normalized != base:
                raise ValueError("Unsafe cache merge: partition configuration mismatch")
            for field in ("source_sha256", "input_manifest", "weather", "noise_version", "numpy_version", "python_version"):
                if manifest[field] != inputs[field]:
                    raise ValueError(f"Unsafe cache merge: {field} differs")
            for line in (part_dir / "checkpoints" / "paths.jsonl").read_text().splitlines():
                record = json.loads(line)
                if record["key"] in seen or record["role"] != "training" or record["seed"] != base["reliability"]["seed"]:
                    raise ValueError("Partition paths overlap or contain wrong sample roles/seeds")
                seen.add(record["key"])
                destination.write(line + "\n")
    write_json(final / "run_manifest.json", dict(run_sha256=digest, cache_provenance=[str(p) for p in completed], **inputs))
    # Cached training only; the first holdout draw occurs AFTER the final full
    # grid planner freezes its incumbent inside the ordinary CLI.
    with (output / "full_grid.log").open("w") as stream:
        result = subprocess.run([sys.executable, "-m", "polar_reliability_planning.recovery_cli", "plan",
                                 "--config", str(full_config_path), "--output", str(final), "--resume"],
                                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    write_json(output / "workflow_summary.json", dict(workers=args.workers, partitions=len(jobs),
               cached_training_paths=len(seen), partition_training_wall_seconds=training_seconds,
               total_wall_seconds=time.perf_counter() - started, exit_code=result.returncode,
               final_result=str(final / "capacity_result.json"),
               scope="same full grid, same training primitives, same frozen library; disjoint capacity partitioning"))
    print((output / "workflow_summary.json").read_text(), flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
