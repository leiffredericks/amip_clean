#!/usr/bin/env python3
"""Create monthly tas--global-N covariance maps from Maria CMIP files.

The calculation reproduces the intent of ``Dave_maps.ipynb`` without relying on
anonymous ensemble indices or persistent monthly intermediates:

1. select March 2000--December 2014;
2. make N = rsdt - rlut - rsut on each model's native grid;
3. remap tas and N to the ERA5 128x64 grid in a temporary directory;
4. remove each calendar month's linear trend;
5. calculate cov(standardised tas, global-mean N).

Only final 2-D maps and a CSV report are retained.  Temporary regridded NetCDF
files are removed after every model/member task, including after a failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr


START = "2000-03-01"
END = "2014-12-31"
DEFAULT_ERA5 = "/rugenstein-archive/senne/data/obs/ERA5/tas_ERA5_128x64_194001-202512.nc"
DEFAULT_CERES = (
    "/rugenstein-archive/senne/data/observations/CERES/"
    "CERES_EBAF-TOA_Ed4.2.1_Subset_1degx1deg_200003-202604.nc"
)


def coordinate_name(data: xr.DataArray, kind: str) -> str:
    choices = ("lat", "latitude") if kind == "lat" else ("lon", "longitude")
    for name in choices:
        if name in data.dims or name in data.coords:
            return name
    raise ValueError(f"Could not find a {kind} coordinate in {tuple(data.dims)}")


def select_months(data: xr.DataArray) -> xr.DataArray:
    selected = data.sel(time=slice(START, END))
    if selected.sizes.get("time") != 178:
        raise ValueError(
            f"Expected 178 monthly records from {START[:7]} through {END[:7]}; "
            f"found {selected.sizes.get('time')}"
        )
    return selected


def monthly_linear_detrend(data: xr.DataArray) -> xr.DataArray:
    """Remove the mean and linear trend separately for each calendar month."""
    if "time" not in data.dims:
        raise ValueError("Monthly detrending requires a time dimension")
    ordered = data.transpose("time", ...)
    values = ordered.values.astype(np.float64, copy=True)
    months = ordered["time"].dt.month.values

    for month in range(1, 13):
        indexes = np.flatnonzero(months == month)
        y = values[indexes]
        x = np.arange(len(indexes), dtype=np.float64)
        x = x - x.mean()
        valid = np.isfinite(y)
        count = valid.sum(axis=0)
        mean = np.nansum(y, axis=0) / np.where(count, count, np.nan)
        centered = y - mean
        denominator = np.nansum(valid * x.reshape((-1,) + (1,) * (y.ndim - 1)) ** 2, axis=0)
        numerator = np.nansum(
            centered * x.reshape((-1,) + (1,) * (y.ndim - 1)), axis=0
        )
        slope = numerator / np.where(denominator, denominator, np.nan)
        values[indexes] = centered - slope * x.reshape((-1,) + (1,) * (y.ndim - 1))

    return xr.DataArray(values, coords=ordered.coords, dims=ordered.dims, attrs=ordered.attrs)


def global_mean(data: xr.DataArray) -> xr.DataArray:
    """Cosine-latitude weighted global mean, preserving the time dimension."""
    lat_name = coordinate_name(data, "lat")
    lon_name = coordinate_name(data, "lon")
    weights = np.cos(np.deg2rad(data[lat_name]))
    return data.weighted(weights).mean((lat_name, lon_name), skipna=True)


def standardised_covariance_map(tas: xr.DataArray, net_flux: xr.DataArray) -> xr.DataArray:
    """Return time-mean tas-standardised covariance with global-mean N."""
    tas_anom = monthly_linear_detrend(select_months(tas))
    n_anom = monthly_linear_detrend(global_mean(select_months(net_flux)))
    tas_std = tas_anom.std("time", skipna=True)
    standardised_tas = tas_anom / tas_std.where(tas_std > 0)
    pattern = (standardised_tas * n_anom).mean("time", skipna=True)
    pattern.name = "tas_N_standardised_covariance"
    pattern.attrs = {
        "long_name": "Monthly detrended covariance of standardised tas with global-mean N",
        "units": net_flux.attrs.get("units", "W m-2"),
        "period": "2000-03 through 2014-12",
        "N_definition": "rsdt - rlut - rsut",
    }
    return pattern


def cdo(command: list[str]) -> None:
    subprocess.run(["cdo", "-L", "-O", *command], check=True)


def remap_observed_n(ceres_path: Path, target_grid: Path, temp_dir: Path, operator: str) -> Path:
    output = temp_dir / "ceres_N_remapped.nc"
    cdo([f"{operator},{target_grid}", "-selyear,2000/2014", str(ceres_path), str(output)])
    return output


def remap_model_fields(task: dict[str, str], maria_root: Path, target_grid: Path, temp_dir: Path,
                       tas_operator: str, flux_operator: str) -> tuple[Path, Path]:
    paths = {name: maria_root / task[f"{name}_path"].lstrip("./") for name in ("tas", "rsdt", "rlut", "rsut")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source file(s): " + "; ".join(missing))

    tas_output = temp_dir / "tas_remapped.nc"
    n_output = temp_dir / "N_remapped.nc"
    cdo([f"{tas_operator},{target_grid}", "-selyear,2000/2014", str(paths["tas"]), str(tas_output)])
    cdo([
        f"{flux_operator},{target_grid}", "-selyear,2000/2014", "-expr,N=rsdt-rlut-rsut",
        "-merge", str(paths["rsdt"]), str(paths["rlut"]), str(paths["rsut"]), str(n_output),
    ])
    return tas_output, n_output


def first_data_variable(dataset: xr.Dataset, preferred: str) -> xr.DataArray:
    if preferred in dataset:
        return dataset[preferred]
    candidates = list(dataset.data_vars)
    if len(candidates) == 1:
        return dataset[candidates[0]]
    raise KeyError(f"Expected variable {preferred!r}; available variables: {candidates}")


def write_map(pattern: xr.DataArray, path: Path, metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = pattern.to_dataset()
    dataset.attrs.update(metadata)
    dataset.to_netcdf(path)


def plot_map(pattern: xr.DataArray, path: Path, title: str) -> None:
    """Optional SVG output; imported only when plotting is requested."""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    lat_name, lon_name = coordinate_name(pattern, "lat"), coordinate_name(pattern, "lon")
    magnitude = float(np.nanmax(np.abs(pattern.values)))
    figure = plt.figure(figsize=(11, 6))
    axis = figure.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=210))
    image = axis.pcolormesh(
        pattern[lon_name], pattern[lat_name], pattern,
        transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=-magnitude, vmax=magnitude, shading="auto",
    )
    axis.coastlines(linewidth=0.6)
    axis.set_global()
    figure.colorbar(image, ax=axis, orientation="horizontal", pad=0.05, shrink=0.7)
    axis.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", transparent=True, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maria-root", type=Path, required=True, help="Remote root containing Maria model folders")
    parser.add_argument("--catalog", type=Path, default=Path("maria_catalog/maria_processing_tasks.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/monthly_mca"))
    parser.add_argument("--era5", type=Path, default=Path(DEFAULT_ERA5))
    parser.add_argument("--ceres", type=Path, default=Path(DEFAULT_CERES))
    parser.add_argument("--experiments", nargs="+", default=["amip-piForcing", "amip-hist", "historical"])
    parser.add_argument("--models", nargs="+", help="Optional model subset for a test run")
    parser.add_argument("--tas-remap", default="remapbil", help="CDO remapping operator for tas")
    parser.add_argument("--flux-remap", default="remapcon", help="CDO remapping operator for N and CERES")
    parser.add_argument("--plot", action="store_true", help="Also write SVG maps (requires cartopy)")
    parser.add_argument("--skip-member-maps", action="store_true", help="Save ensemble means only")
    args = parser.parse_args()

    with args.catalog.open(encoding="utf-8") as handle:
        tasks = json.load(handle)
    tasks = [
        task for task in tasks
        if task["experiment"] in args.experiments and (not args.models or task["model"] in args.models)
    ]
    if not tasks:
        raise SystemExit("No complete catalog tasks matched the requested models and experiments.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    era5 = xr.open_dataset(args.era5)
    observed_tas = first_data_variable(era5, "tas") if "tas" in era5 else first_data_variable(era5, "t2m")
    observed_tas = select_months(observed_tas)
    ensemble_sums: dict[tuple[str, str], xr.DataArray] = {}
    ensemble_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    report: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="monthly_mca_") as scratch:
        scratch_dir = Path(scratch)
        observed_n_path = remap_observed_n(args.ceres, args.era5, scratch_dir, args.flux_remap)
        with xr.open_dataset(observed_n_path) as observed_n_ds:
            observed_n = first_data_variable(observed_n_ds, "toa_net_all_mon").load()
        obs_pattern = standardised_covariance_map(observed_tas, observed_n)
        write_map(obs_pattern, args.output_dir / "observation_mca_map.nc", {"source": "ERA5 tas and CERES EBAF-TOA"})
        if args.plot:
            plot_map(obs_pattern, args.output_dir / "observation_mca_map.svg", "Observations: tas vs global N")

        for task in tasks:
            label = f"{task['experiment']} / {task['model']} / {task['member']}"
            try:
                with tempfile.TemporaryDirectory(dir=scratch_dir, prefix="task_") as task_scratch:
                    tas_path, n_path = remap_model_fields(
                        task, args.maria_root, args.era5, Path(task_scratch), args.tas_remap, args.flux_remap
                    )
                    with xr.open_dataset(tas_path) as tas_ds, xr.open_dataset(n_path) as n_ds:
                        pattern = standardised_covariance_map(
                            first_data_variable(tas_ds, "tas").load(), first_data_variable(n_ds, "N").load()
                        )
                key = (task["experiment"], task["model"])
                ensemble_sums[key] = pattern if key not in ensemble_sums else ensemble_sums[key] + pattern
                ensemble_counts[key] += 1
                map_path = args.output_dir / "members" / task["experiment"] / task["model"] / f"{task['member']}.nc"
                if not args.skip_member_maps:
                    write_map(pattern, map_path, task)
                    if args.plot:
                        plot_map(pattern, map_path.with_suffix(".svg"), label)
                report.append({**task, "result": "success", "message": ""})
                print(f"complete: {label}")
            except Exception as error:  # Continue through independent model/member tasks.
                report.append({**task, "result": "failed", "message": str(error)})
                print(f"failed: {label}: {error}")

    for (experiment, model), total in ensemble_sums.items():
        mean_map = total / ensemble_counts[(experiment, model)]
        mean_path = args.output_dir / "ensemble_means" / experiment / f"{model}.nc"
        metadata = {
            "model": model,
            "experiment": experiment,
            "ensemble_member_count": str(ensemble_counts[(experiment, model)]),
        }
        write_map(mean_map, mean_path, metadata)
        if args.plot:
            plot_map(mean_map, mean_path.with_suffix(".svg"), f"{experiment}: {model} ensemble mean")

    fieldnames = list(report[0]) if report else ["result", "message"]
    with (args.output_dir / "run_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report)


if __name__ == "__main__":
    main()
