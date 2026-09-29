"""Plain-dictionary helpers for notebook diagnostics; no scientific calculations hidden here."""
import numpy as np
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timezone
import os
import tempfile


def setup_diagnostic_archive(quantity_info, model_names, member_ids, *,
                             experiments=("historical", "historical_matched", "amip_hist"),
                             coordinates=None, mask=None, metadata=None,
                             save=False, savepath=None, overwrite=False):
    """Create empty member/model/MMM dictionaries and copy the selected labels.

    quantity_info maps your field names to dictionaries of units/meaning/etc.
    Optional field entries: shape (per member), dims (excluding member), and
    apply_mask=True (use the supplied 2-D mask on the final lat/lon axes).
    Unspecified shapes are learned from the first stored model for each field.
    save/savepath/overwrite are settings ONLY: setup performs no disk writes.
    """
    experiments = list(experiments)
    if not experiments or len(experiments) != len(set(experiments)):
        raise ValueError("experiments must be nonempty and unique.")
    if not quantity_info or any(not isinstance(q, str) or not q.strip() for q in quantity_info):
        raise ValueError("Supply a nonempty quantity_info dictionary with meaningful string keys.")
    info, coords = deepcopy(quantity_info), deepcopy(coordinates or {})
    names, ids, shapes = {}, {}, {}
    for quantity, description in info.items():
        if not isinstance(description, dict):
            raise ValueError(f"{quantity}: metadata must be a dictionary.")
        shape = description.get("shape")
        if shape is not None:
            shape = tuple(shape)
            if any(not isinstance(n, (int, np.integer)) or n < 1 for n in shape):
                raise ValueError(f"{quantity}: shape must contain positive integers; () denotes a scalar.")
        shapes[quantity] = shape
        if description.get("dims") is not None:
            description["dims"] = tuple(description["dims"])
    for experiment in experiments:
        names[experiment] = list(model_names[experiment])
        if not names[experiment] or len(names[experiment]) != len(set(names[experiment])):
            raise ValueError(f"{experiment}: model names must be nonempty and unique.")
        ids[experiment] = {}
        for model in names[experiment]:
            labels = np.asarray(member_ids[experiment][model])
            if labels.ndim != 1 or not len(labels) or len(set(labels.tolist())) != len(labels):
                raise ValueError(f"{experiment}/{model}: supply nonempty, unique 1-D member labels.")
            ids[experiment][model] = labels.copy()

    # A mask is an inclusion rule, not a fractional area weight. It is stored
    # here, but applied only by store_diagnostic_members for opt-in fields.
    selected_mask = None
    if mask is not None:
        supplied = np.ma.asarray(mask, dtype=float).filled(np.nan)
        if supplied.ndim != 2:
            raise ValueError("mask must be a 2-D inclusion mask, or None.")
        selected_mask = np.isfinite(supplied) & (supplied != 0)
        if "lat" in coords and "lon" in coords and selected_mask.shape != (len(coords["lat"]), len(coords["lon"])):
            raise ValueError("mask must match the supplied latitude/longitude grid.")

    # None means NOT YET COMPUTED, rather than a valid zero or an all-NaN result.
    # A field can be a scalar, series, map, matrix, or higher-dimensional array.
    empty_summary = lambda: {e: {q: None for q in info} for e in experiments}
    return {
        "schema": "member_diagnostic_archive", "schema_version": 1,
        "experiments": experiments, "quantity_info": info, "field_shapes": shapes,
        "member_values": {e: {q: {m: None for m in names[e]} for q in info} for e in experiments},
        "model_means": empty_summary(), "MMM": empty_summary(),
        "model_n_valid": empty_summary(), "MMM_n_valid": empty_summary(),
        "model_names": names, "member_ids": ids, "coordinates": coords, "mask": selected_mask,
        "settings": {"save": bool(save), "savepath": None if savepath is None else str(savepath),
                     "overwrite": bool(overwrite)},
        "aggregated": False,
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(), "user": deepcopy(metadata or {}),
            "aggregation": "Finite arithmetic member means, then equal-model means, entry by entry",
            "missing": "NaN/inf omitted from means; valid counts retained; None means not computed",
            "mask_rule": "Finite nonzero includes; applied only to final two axes of opt-in fields",
            "alignment": "No regridding, time alignment, mode matching, or member pairing is inferred",
            "reused_from": {},
        },
    }


def _checked_member_array(archive, experiment, quantity, model, values):
    """Check one completed model's member array without altering its values."""
    # Looking up this slot also checks experiment/quantity/model membership.
    archive["member_values"][experiment][quantity][model]
    array = np.ma.asarray(values)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{experiment}/{model}/{quantity}: supply real numeric values, not objects/strings/complex data.")
    array = np.ma.asarray(array, dtype=float).filled(np.nan)
    n_members = len(archive["member_ids"][experiment][model])
    if array.ndim < 1 or array.shape[0] != n_members or any(n == 0 for n in array.shape):
        raise ValueError(f"{experiment}/{model}/{quantity}: expected member axis of length {n_members}; got {array.shape}.")
    shape, expected = array.shape[1:], archive["field_shapes"][quantity]
    if expected is not None and shape != expected:
        raise ValueError(f"{quantity}: per-member shape {shape} differs from registered/inferred {expected}.")
    dims = archive["quantity_info"][quantity].get("dims")
    if dims is not None:
        if len(dims) != len(shape):
            raise ValueError(f"{quantity}: dims must describe the PER-MEMBER axes {shape}.")
        for dimension, size in zip(dims, shape):
            if dimension in archive["coordinates"]:
                coord = np.asarray(archive["coordinates"][dimension])
                if coord.ndim != 1 or len(coord) != size:
                    raise ValueError(f"{quantity}: dimension {dimension} does not match its saved coordinate.")
    return array


def store_diagnostic_members(archive, experiment, quantity, model, values):
    """Store (member, ...) values, checking labels/shapes and applying an opted-in mask.

    The supplied member order must match archive['member_ids'][experiment][model].
    This does not calculate the diagnostic or mask any inputs to that calculation.
    Updating values invalidates summaries; aggregate again before saving.
    """
    array = _checked_member_array(archive, experiment, quantity, model, values)
    description, mask = archive["quantity_info"][quantity], archive["mask"]
    if description.get("apply_mask", False) and mask is not None:
        dims = description.get("dims")
        if array.ndim < 3 or array.shape[-2:] != mask.shape:
            raise ValueError(f"{quantity}: masked fields must end in the mask's (lat, lon) shape.")
        if dims is not None and tuple(dims[-2:]) != ("lat", "lon"):
            raise ValueError(f"{quantity}: apply_mask requires final dimensions ('lat', 'lon').")
        array = np.where(mask, array, np.nan)
    else:
        array = array.copy()       # Later edits to stored data must not alter source arrays.
    archive["member_values"][experiment][quantity][model] = array
    archive["field_shapes"][quantity] = array.shape[1:]
    archive["aggregated"] = False
    # Clear all means to prevent inadvertently inspecting stale aggregated results.
    for key in ["model_means", "MMM", "model_n_valid", "MMM_n_valid"]:
        for e in archive["experiments"]:
            for q in archive["quantity_info"]:
                archive[key][e][q] = None


def copy_diagnostic_subset(archive, source="historical", target="historical_matched", *, overwrite=False):
    """Reuse a subset's member diagnostics without rerunning their calculations.

    Models must exist in source. Members are selected/reordered by explicit IDs,
    not file positions. Copies are independent arrays. Appropriate only when the
    diagnostic itself does not change with the selected experiment/model group.
    """
    if source == target:
        raise ValueError("source and target must be different experiments.")
    selections = {}
    # Check every requested slot before copying, so missing IDs/results do not
    # leave a partially copied target. A populated target needs explicit overwrite.
    for model in archive["model_names"][target]:
        lookup = {label: i for i, label in enumerate(archive["member_ids"][source][model].tolist())}
        labels = archive["member_ids"][target][model].tolist()
        if any(label not in lookup for label in labels):
            raise ValueError(f"{target}/{model}: a member ID is absent from {source}.")
        selections[model] = [lookup[label] for label in labels]
        for quantity in archive["quantity_info"]:
            values = archive["member_values"][source][quantity][model]
            if values is None:
                raise ValueError(f"Calculate {source}/{model}/{quantity} before copying.")
            _checked_member_array(archive, source, quantity, model, values)
            if archive["member_values"][target][quantity][model] is not None and not overwrite:
                raise ValueError(f"{target}/{model}/{quantity} already populated; set overwrite=True deliberately.")
    for model, positions in selections.items():
        for quantity in archive["quantity_info"]:
            values = archive["member_values"][source][quantity][model][positions]
            store_diagnostic_members(archive, target, quantity, model, values)
    archive["metadata"]["reused_from"][target] = source


def _finite_diagnostic_mean(values):
    """Finite mean/count along the first axis; works for scalars through ND arrays."""
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    total = np.where(finite, values, 0.).sum(axis=0)
    mean = np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)
    return mean, count


def aggregate_diagnostic_archive(archive):
    """Average completed member diagnostics within models, then equally across models.

    These are literal arithmetic means of YOUR calculated quantities. No Fisher-z,
    squaring, root, pooling, or variance estimation is automatically performed.
    All-NaN results are allowed (with zero counts); uncomputed None slots are not.
    """
    archive["aggregated"] = False
    missing = [(e, q, m) for e in archive["experiments"] for q in archive["quantity_info"]
               for m in archive["model_names"][e] if archive["member_values"][e][q][m] is None]
    if missing:
        raise ValueError(f"{len(missing)} model/quantity slots are uncomputed. First entries: {missing[:5]}")
    summaries = {key: {e: {} for e in archive["experiments"]}
                 for key in ["model_means", "MMM", "model_n_valid", "MMM_n_valid"]}
    for experiment in archive["experiments"]:
        for quantity in archive["quantity_info"]:
            means, counts = [], []
            for model in archive["model_names"][experiment]:
                values = _checked_member_array(archive, experiment, quantity, model,
                                                archive["member_values"][experiment][quantity][model])
                mean, count = _finite_diagnostic_mean(values)
                means.append(mean)
                counts.append(count)
            # Stack in the explicit model order. No model gains weight merely
            # by having more members. Counts vary entry-by-entry when data are missing.
            stack = np.stack(means, axis=0)
            summaries["model_means"][experiment][quantity] = stack
            summaries["model_n_valid"][experiment][quantity] = np.stack(counts, axis=0)
            summaries["MMM"][experiment][quantity], summaries["MMM_n_valid"][experiment][quantity] = _finite_diagnostic_mean(stack)
    archive.update(summaries)
    archive["aggregated"] = True
    archive["metadata"]["aggregated_utc"] = datetime.now(timezone.utc).isoformat()
    return archive


def _check_completed_diagnostic_archive(archive):
    """Check schema, completion, member counts, and summary shapes, not scientific correctness."""
    if not isinstance(archive, dict) or archive.get("schema") != "member_diagnostic_archive" or archive.get("schema_version") != 1:
        raise ValueError("Not a supported member diagnostic archive.")
    if not archive.get("aggregated", False):
        raise ValueError("Aggregate completed diagnostics before saving/loading this archive.")
    for experiment in archive["experiments"]:
        for quantity in archive["quantity_info"]:
            shape = archive["field_shapes"][quantity]
            if shape is None:
                raise ValueError(f"{quantity}: no per-member shape registered.")
            for model in archive["model_names"][experiment]:
                _checked_member_array(archive, experiment, quantity, model,
                                      archive["member_values"][experiment][quantity][model])
            for key in ["model_means", "model_n_valid", "MMM", "MMM_n_valid"]:
                expected = (len(archive["model_names"][experiment]),) + shape if key.startswith("model_") else shape
                if np.shape(archive[key][experiment][quantity]) != expected:
                    raise ValueError(f"{key}/{experiment}/{quantity}: invalid summary shape.")


def save_diagnostic_archive(archive, *, save=None, savepath=None, overwrite=None):
    """Optionally save one self-contained .npy file; disabled unless save=True.

    Defaults come from setup. Existing targets are protected unless overwrite=True.
    Write a temporary file in the same directory, then publish only after success;
    an interrupted/failed write does not replace an existing completed archive.
    Use store_diagnostic_members to update values, and rerun aggregation after
    ANY direct array edits. This saver checks structure, not numerical freshness.
    """
    settings = archive["settings"]
    enabled = settings["save"] if save is None else save
    if not enabled:
        print("Save disabled; results remain in memory.")
        return None
    _check_completed_diagnostic_archive(archive)
    destination = settings["savepath"] if savepath is None else savepath
    if destination is None:
        raise ValueError("Provide savepath ending in .npy when saving.")
    path = Path(destination).expanduser()
    replace = settings["overwrite"] if overwrite is None else overwrite
    if path.suffix != ".npy":
        raise ValueError("savepath must be a .npy FILE, not a directory.")
    if path.exists() and not replace:
        raise FileExistsError(f"Archive already exists: {path}. Choose a new name or set overwrite=True.")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(archive)
    payload["metadata"] = {**archive["metadata"], "saved_utc": datetime.now(timezone.utc).isoformat()}
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}_", suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            np.save(handle, payload, allow_pickle=True)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary_path, path)
        else:
            # Same-filesystem hard link publishes the finished file without any
            # overwrite race. If another file appeared meanwhile, this raises.
            os.link(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()       # Remove only this call's exact temporary file.
    archive["metadata"]["saved_utc"] = payload["metadata"]["saved_utc"]
    print(f"Saved diagnostic archive: {path}")
    return path


def load_diagnostic_archive(savepath):
    """Load a completed archive saved above. ONLY load files you created/trust.

    .npy dictionaries use pickle, which is not safe for untrusted files. Checks
    after loading verify structure; they cannot make untrusted pickle safe.
    Loading does not recompute means or change existing notebook variables.
    """
    archive = np.load(Path(savepath).expanduser(), allow_pickle=True).item()
    _check_completed_diagnostic_archive(archive)
    return archive
