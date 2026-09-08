# %% Weighted comparison of two maps: x is the reference, y is the comparison
import numpy as np


def compare_maps(x, y, weights=None, *, mask=None):
    """Score one pair of aligned maps of any identical shape; return a dictionary.

    x is the target/reference; y is the comparison. Bias is y minus x, and
    amplitude ratios are y divided by x. Both maps must have the same units.
    weights are cell areas or proportional weights broadcastable to the map shape;
    None gives equal-cell weighting. mask is optional, with True meaning include.
    Masked/nonfinite data and nonpositive/nonfinite weights are excluded jointly.
    Negative finite weights raise. No valid paired cells raises.

    Coordinates are not used or aligned: put both maps on the same ordered grid
    first. Flattened, rectilinear, curvilinear, and unstructured grids all work.
    For (lat, lon) maps, latitude-only weights must be shaped (lat, 1).
    All dimensions are reduced: pass one map per call, not a stack of members.

    Spatial moments use normalized area weights, not a sample-ddof correction.
    Undefined correlations or ratios return NaN; no epsilon is added to divisors.
    lin_ccc uses these same weighted moments. Identical constant maps give NaN
    (zero denominator); unequal constants or one constant map give CCC = 0.
    """
    # Preserve masked-array exclusions, and require point-for-point matching shapes.
    # Equal shapes alone cannot verify that geographic coordinates are aligned.
    x = np.ma.asarray(x, dtype=float).filled(np.nan)
    y = np.ma.asarray(y, dtype=float).filled(np.nan)
    if x.ndim == 0 or x.shape != y.shape:
        raise ValueError("x and y must have identical, nonscalar map shapes.")
    weights = 1.0 if weights is None else weights
    weights = np.ma.asarray(weights, dtype=float).filled(np.nan)
    weights = np.broadcast_to(weights, x.shape)
    if np.any(np.isfinite(weights) & (weights < 0)):
        raise ValueError("Area weights cannot be negative.")
    include = True if mask is None else np.ma.asarray(mask, dtype=bool).filled(False)
    include = np.broadcast_to(include, x.shape)

    # Every statistic uses exactly the same paired support and renormalized weights.
    # Coverage measures retained area relative to the requested, positive-weight domain.
    support = include & np.isfinite(weights) & (weights > 0)
    valid = support & np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        raise ValueError("No finite paired values with positive weights remain.")
    scaled_weights = weights[support] / weights[support].max()
    w = weights[valid] / weights[support].max()
    valid_weight_fraction = float(w.sum() / scaled_weights.sum())
    w = w / w.sum()
    xv, yv = x[valid], y[valid]

    # Spatial means and contrasts: remove each map's OWN area-weighted mean.
    # Handle exactly constant maps explicitly to avoid spurious roundoff contrast.
    mean_x = float(xv[0]) if np.all(xv == xv[0]) else float(w @ xv)
    mean_y = float(yv[0]) if np.all(yv == yv[0]) else float(w @ yv)
    xc, yc = xv - mean_x, yv - mean_y
    std_x, std_y = np.sqrt(w @ xc**2), np.sqrt(w @ yc**2)
    rms_x, rms_y = np.sqrt(w @ xv**2), np.sqrt(w @ yv**2)

    # Physical-unit discrepancies: total RMSE includes the mean offset, while
    # centered RMSE compares spatial departures from the two respective means.
    difference = yv - xv
    bias = float(w @ difference)
    rmse = float(np.sqrt(w @ difference**2))
    centered_rmse = float(np.sqrt(w @ (yc - xc)**2))

    # Correlation/contrast describe centered patterns; cosine/norm retain offsets.
    # A tiny positive denominator is not masked automatically: inspect std_x/rms_x.
    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator > 0 else np.nan

    covariance = float(w @ (xc * yc))
    correlation = ratio(covariance, std_x * std_y)
    cosine = ratio(w @ (xv * yv), rms_x * rms_y)
    contrast_ratio, norm_ratio = ratio(std_y, std_x), ratio(rms_y, rms_x)

    # Lin's CCC measures agreement with y=x, penalizing both mean and contrast
    # differences as well as imperfect correlation. Use covariance directly so
    # a constant map can give zero CCC even though Pearson correlation is undefined.
    ccc_denominator = std_x**2 + std_y**2 + (mean_y - mean_x)**2
    lin_ccc = ratio(2 * covariance, ccc_denominator)

    # Keep physical and reference-normalized errors under distinct names.
    # The weighted RMS ratio equals the weighted L2-norm ratio on common support.
    return {
        "rmse": rmse, "bias": bias, "centered_rmse": centered_rmse,
        "pattern_correlation": float(np.clip(correlation, -1, 1)),
        "contrast_ratio": contrast_ratio, "norm_ratio": norm_ratio,
        "cosine": float(np.clip(cosine, -1, 1)),
        "lin_ccc": float(np.clip(lin_ccc, -1, 1)),
        "rmse_over_rms_x": ratio(rmse, rms_x), "bias_over_rms_x": ratio(bias, rms_x),
        "rmse_over_std_x": ratio(rmse, std_x), "bias_over_std_x": ratio(bias, std_x),
        "centered_rmse_over_std_x": ratio(centered_rmse, std_x),
        "mean_x": mean_x, "mean_y": mean_y,
        "std_x": float(std_x), "std_y": float(std_y),
        "rms_x": float(rms_x), "rms_y": float(rms_y),
        "n_valid": int(valid.sum()), "valid_weight_fraction": valid_weight_fraction,
    }

# %%
