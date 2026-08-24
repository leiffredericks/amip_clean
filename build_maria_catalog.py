#!/usr/bin/env python3
"""Build a processing catalog from Maria's CMIP-style file manifest.

The catalog is deliberately path-based: it reads only a text manifest and never
opens a NetCDF file.  A task is eligible for spatial-N processing only when one
unambiguous file exists for every required variable for a given
model/experiment/member tuple.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict


DEFAULT_REQUIRED = ("tas", "rsdt", "rlut", "rsut")
MEMBER_PATTERN = re.compile(r"^r\d+i\d+p\d+(?:f\d+)?$")


def parse_file_path(line: str) -> dict[str, str] | None:
    """Parse one CMIP-style manifest path, returning None for directories.

    Expected layout: ./<model>/<experiment>/<var>_<table>_<model>_<experiment>
    _<member>_<grid>_<period>.nc.  The path components are used as the
    authoritative model and experiment names, which avoids brittle assumptions
    about hyphens in their names.
    """
    path = line.strip()
    if not path.endswith(".nc"):
        return None

    parts = Path(path).parts
    # For './Model/experiment/file.nc', pathlib may retain '.' as a part.
    parts = tuple(part for part in parts if part not in (".", "/"))
    if len(parts) < 3:
        return None
    model, experiment, filename = parts[-3:]

    stem = Path(filename).stem
    tokens = stem.split("_")
    if len(tokens) != 7:
        return None
    variable, table, filename_model, filename_experiment, member, grid, period = tokens
    if not MEMBER_PATTERN.fullmatch(member):
        return None
    if not re.fullmatch(r"\d{6}-\d{6}", period):
        return None

    return {
        "path": path,
        "model": model,
        "experiment": experiment,
        "member": member,
        "variable": variable,
        "table": table,
        "grid": grid,
        "period": period,
        "filename_model": filename_model,
        "filename_experiment": filename_experiment,
    }


def catalog_records(
    manifest: Path, required: tuple[str, ...], experiment_filter: set[str] | None
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Return one status record per observed model/experiment/member tuple."""
    files: DefaultDict[tuple[str, str, str], DefaultDict[str, list[dict[str, str]]]]
    files = defaultdict(lambda: defaultdict(list))
    stats = {"lines": 0, "netcdf_paths": 0, "parsed_cmip_paths": 0, "relevant_paths": 0}

    with manifest.open(encoding="utf-8") as handle:
        for raw_line in handle:
            stats["lines"] += 1
            parsed = parse_file_path(raw_line)
            if parsed is None:
                continue
            stats["netcdf_paths"] += 1
            stats["parsed_cmip_paths"] += 1
            if parsed["variable"] not in required:
                continue
            if experiment_filter and parsed["experiment"] not in experiment_filter:
                continue
            stats["relevant_paths"] += 1
            key = (parsed["model"], parsed["experiment"], parsed["member"])
            files[key][parsed["variable"]].append(parsed)

    records: list[dict[str, str]] = []
    for model, experiment, member in sorted(files):
        candidates = files[(model, experiment, member)]
        missing = [var for var in required if not candidates[var]]
        ambiguous = [var for var in required if len(candidates[var]) > 1]
        status = "complete" if not missing and not ambiguous else "incomplete"
        if ambiguous:
            status = "ambiguous"

        record = {
            "model": model,
            "experiment": experiment,
            "member": member,
            "status": status,
            "missing_variables": ";".join(missing),
            "ambiguous_variables": ";".join(ambiguous),
        }
        for var in required:
            choices = candidates[var]
            record[f"{var}_path"] = ";".join(item["path"] for item in choices)
            record[f"{var}_table"] = ";".join(item["table"] for item in choices)
            record[f"{var}_grid"] = ";".join(item["grid"] for item in choices)
            record[f"{var}_period"] = ";".join(item["period"] for item in choices)
        records.append(record)
    return records, stats


def write_outputs(records: list[dict[str, str]], stats: dict[str, int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0]) if records else ["model", "experiment", "member", "status"]

    catalog_path = output_dir / "maria_catalog.csv"
    with catalog_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    complete = [record for record in records if record["status"] == "complete"]
    issues = [record for record in records if record["status"] != "complete"]
    with (output_dir / "maria_processing_tasks.json").open("w", encoding="utf-8") as handle:
        json.dump(complete, handle, indent=2)
        handle.write("\n")
    with (output_dir / "maria_missing_or_ambiguous.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(issues)

    by_status: DefaultDict[str, int] = defaultdict(int)
    for record in records:
        by_status[record["status"]] += 1
    summary = {
        "input": stats,
        "tuple_count": len(records),
        "complete_task_count": len(complete),
        "status_counts": dict(sorted(by_status.items())),
        "outputs": {
            "catalog": str(catalog_path),
            "processing_tasks": str(output_dir / "maria_processing_tasks.json"),
            "issues": str(output_dir / "maria_missing_or_ambiguous.csv"),
        },
    }
    with (output_dir / "maria_catalog_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Cataloged {len(records)} model/experiment/member tuples.")
    print(f"Complete processing tasks: {len(complete)}")
    print(f"Incomplete or ambiguous tuples: {len(issues)}")
    for name, path in summary["outputs"].items():
        print(f"{name}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Maria path manifest produced with find")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("maria_catalog"), help="Directory for catalog files"
    )
    parser.add_argument(
        "--required-vars", nargs="+", default=list(DEFAULT_REQUIRED), help="Variables required for a task"
    )
    parser.add_argument(
        "--experiments", nargs="+", help="Optional experiment names to include, e.g. amip-hist"
    )
    args = parser.parse_args()
    required = tuple(args.required_vars)
    records, stats = catalog_records(args.manifest, required, set(args.experiments or []))
    write_outputs(records, stats, args.output_dir)


if __name__ == "__main__":
    main()
