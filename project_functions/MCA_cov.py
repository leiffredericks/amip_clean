import numpy as np


def field_scalar_covariance(X, y, standardize_X=True, standardize_y=False, ddof=1):
    """Covariance map with optional local X and/or global y standardization.

    X: (time, spatial dimensions...); y: (same time,).
    Only temporal means are removed here; align dates and preprocess first.
    The default reproduces m_i from standardized_temperature_covariance.
    With neither standardized: sigma_X * sigma_y * rho; X only: sigma_y * rho;
    y only: sigma_X * rho; both: rho. Map outputs retain X's spatial shape.
    Missing X excludes that location for the window; missing y raises.
    """
    # Flatten the spatial dimensions so each column is one local time series.
    # Masked entries become NaN; any xarray coordinate labels are not retained.
    X, y = np.ma.asarray(X, dtype=float).filled(np.nan), np.ma.asarray(y, dtype=float).filled(np.nan)
    if X.ndim < 2 or y.ndim != 1 or X.shape[0] != y.size:
        raise ValueError("Use X=(time, spatial dimensions...) and y=(same time,).")
    n, spatial_shape = X.shape[0], X.shape[1:]
    if not isinstance(ddof, (int, np.integer)) or not 0 <= ddof < n or n < 2:
        raise ValueError("Need at least two times and an integer 0 <= ddof < n.")
    if not np.isfinite(y).all():
        raise ValueError("y contains missing values; select a common time window first.")
    field, denominator = X.reshape(n, -1), n - ddof

    # Estimate the original amplitudes and covariance using all months at each
    # complete location. These reference quantities do not depend on the switches.
    complete = np.isfinite(field).all(axis=0)
    anomalies, y_anomaly = field[:, complete] - field[:, complete].mean(axis=0), y - y.mean()
    sigma_X, raw_covariance = np.full(field.shape[1], np.nan), np.full(field.shape[1], np.nan)
    sigma_X[complete] = np.sqrt(np.sum(anomalies**2, axis=0) / denominator)
    sigma_y = np.sqrt(y_anomaly @ y_anomaly / denominator)
    raw_covariance[complete] = anomalies.T @ y_anomaly / denominator
    rho = np.full(field.shape[1], np.nan)
    varying = complete & (sigma_X > 0)
    if sigma_y > 0:
        rho[varying] = raw_covariance[varying] / (sigma_X[varying] * sigma_y)

    # A constant X series has zero raw covariance but cannot be standardized.
    # A constant y is allowed unless y standardization is requested; rho is NaN.
    valid = varying if standardize_X else complete
    if not valid.any():
        raise ValueError("No valid locations remain for the chosen standardization.")
    if standardize_y and sigma_y == 0:
        raise ValueError("Cannot standardize y because its temporal SD is zero.")

    # Apply the selected scaling to the anomalies, then take their covariance.
    # No spatial weights enter this local map. Standardization changes its units.
    X_analysis = field[:, valid] - field[:, valid].mean(axis=0)
    if standardize_X:
        X_analysis = X_analysis / sigma_X[valid]
    y_analysis = y_anomaly / sigma_y if standardize_y else y_anomaly
    covariance_map = np.full(field.shape[1], np.nan)
    covariance_map[valid] = X_analysis.T @ y_analysis / denominator

    # Restore map shapes and retain both switches so the chosen definition is explicit.
    return {"covariance_map": covariance_map.reshape(spatial_shape), "rho": rho.reshape(spatial_shape),
            "sigma_X": sigma_X.reshape(spatial_shape), "sigma_y": sigma_y,
            "raw_covariance": raw_covariance.reshape(spatial_shape), "valid_mask": valid.reshape(spatial_shape),
            "standardize_X": standardize_X, "standardize_y": standardize_y, "n_time": n, "ddof": ddof}


import numpy as np


def field_mca(X, Y, weights_X=None, weights_Y=None, k=None,
              standardize_X=True, standardize_Y=False, ddof=1):
    """MCA of K = sqrt(D_X) @ Cov(X_analysis, Y_analysis) @ sqrt(D_Y).

    Inputs have time first: (time, space) or (time, lat, lon, ...).
    Weights must match each field's spatial shape; None means equal cell weights.
    Defaults match our temperature/radiation choice: standardize X only.
    Set both standardize flags False for ordinary physical-covariance MCA.
    Supply aligned, preprocessed anomalies; only temporal means are removed here.
    Locations with any missing value, zero SD, or nonpositive/nonfinite weight
    are excluded for the whole window. No time rows are silently dropped.
    Patterns/maps have shape (*spatial_shape, mode); scores have (time, mode).
    """
    # Flatten the maps while retaining their shapes for restoring geographical output.
    X, Y = np.ma.asarray(X, dtype=float).filled(np.nan), np.ma.asarray(Y, dtype=float).filled(np.nan)
    if X.ndim < 2 or Y.ndim < 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y need matching time axes followed by spatial dimensions.")
    n, shape_X, shape_Y = X.shape[0], X.shape[1:], Y.shape[1:]
    if not isinstance(ddof, (int, np.integer)) or not 0 <= ddof < n or n < 2:
        raise ValueError("Need at least two times and an integer 0 <= ddof < n.")
    denominator = n - ddof

    # Apply the same preparation to each field. Keep original anomalies for loadings
    # in physical units, and a separate analysis field for the chosen normalization.
    prepared = []
    for field, weights, standardize in [(X, weights_X, standardize_X), (Y, weights_Y, standardize_Y)]:
        spatial_shape, flat = field.shape[1:], field.reshape(n, -1)
        weights = np.ones(spatial_shape) if weights is None else np.ma.asarray(weights, dtype=float).filled(np.nan)
        if weights.shape != spatial_shape:
            raise ValueError("Each weight array must match its field's spatial shape.")
        weights = weights.ravel()
        valid = np.isfinite(flat).all(axis=0) & np.isfinite(weights) & (weights > 0)
        raw = flat[:, valid] - flat[:, valid].mean(axis=0)
        local_sd = np.sqrt(np.sum(raw**2, axis=0) / denominator)
        varying = local_sd > 0
        valid[np.flatnonzero(valid)[~varying]] = False
        raw, local_sd = raw[:, varying], local_sd[varying]
        if not valid.any():
            raise ValueError("A field has no complete, nonconstant, positively weighted locations.")
        w = weights[valid] / weights[valid].sum()
        analysis = raw / local_sd if standardize else raw.copy()
        prepared.append((raw, local_sd, analysis, w, valid))
    raw_X, sigma_X, Xa, wx, valid_X = prepared[0]
    raw_Y, sigma_Y, Ya, wy, valid_Y = prepared[1]

    # Square-root area weights put the physical weighted inner product into ordinary
    # Euclidean coordinates. Full weights here would count cell area twice.
    Xw, Yw = Xa * np.sqrt(wx), Ya * np.sqrt(wy)

    # The direct formula is K = Xw.T @ Yw / denominator, but a global K can be huge.
    # QR writes Xw.T = Qx @ Rx and Yw.T = Qy @ Ry, giving K = Qx @ core @ Qy.T.
    # Qx and Qy are orthonormal, so decomposing this at-most-time-by-time core and
    # mapping its vectors back gives the SAME spatial SVD, without forming global K.
    Qx, Rx = np.linalg.qr(Xw.T, mode="reduced")
    Qy, Ry = np.linalg.qr(Yw.T, mode="reduced")
    core = Rx @ Ry.T / denominator
    left_core, singular_values, right_core_T = np.linalg.svd(core, full_matrices=False)

    # Keep only numerically supported modes. SCF uses the full spectrum in its
    # denominator, even when we request just the first k modes for interpretation.
    tolerance = np.finfo(float).eps * max(n, Xw.shape[1], Yw.shape[1]) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    k = rank if k is None else k
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= rank:
        raise ValueError(f"k must be between 1 and the numerical cross-covariance rank ({rank}).")
    s, scf = singular_values[:k], singular_values[:k]**2 / np.sum(singular_values**2)
    U, V = Qx @ left_core[:, :k], Qy @ right_core_T[:k].T

    # Remove the coordinate weights for maps, then project to get expansion series.
    # E_X.T @ D_X @ E_X = I; A.T @ B / denominator = diag(s).
    E_X, E_Y = U / np.sqrt(wx)[:, None], V / np.sqrt(wy)[:, None]
    A, B = Xw @ U, Yw @ V
    sd_A = np.sqrt(np.sum(A**2, axis=0) / denominator)
    sd_B = np.sqrt(np.sum(B**2, axis=0) / denominator)
    A_unit, B_unit = A / sd_A, B / sd_B

    # Homogeneous = a field with its OWN expansion series; heterogeneous = with the
    # OTHER field's series. Here "loading" means regression on a unit-SD series,
    # computed from ORIGINAL anomalies, so X loadings remain in K if X was in K.
    # These are single-predictor regressions, not a simultaneous regression on all PCs.
    result = {"singular_values": s, "singular_values_all": singular_values,
              "squared_covariance_fraction": scf, "numerical_rank": rank,
              "U": U, "V": V, "scores_X": A, "scores_Y": B,
              "scores_X_standardized": A_unit, "scores_Y_standardized": B_unit,
              "paired_correlation": s / (sd_A * sd_B), "n_time": n, "ddof": ddof,
              "standardize_X": standardize_X, "standardize_Y": standardize_Y}
    for name, raw, analysis, sigma, w, valid, shape, patterns, own, other, other_raw in [
        ("X", raw_X, Xa, sigma_X, wx, valid_X, shape_X, E_X, A_unit, B_unit, B),
        ("Y", raw_Y, Ya, sigma_Y, wy, valid_Y, shape_Y, E_Y, B_unit, A_unit, A)]:
        homogeneous = raw.T @ own / denominator
        heterogeneous = raw.T @ other / denominator
        maps = {"patterns": patterns, "homogeneous_loadings": homogeneous,
                "heterogeneous_loadings": heterogeneous,
                "homogeneous_correlation": homogeneous / sigma[:, None],
                "heterogeneous_correlation": heterogeneous / sigma[:, None],
                "heterogeneous_covariance_analysis": analysis.T @ other_raw / denominator}

        # Restore the original grid, with NaN at excluded locations and mode last.
        # The analysis covariance map equals singular_value * analysis pattern.
        for label, values in maps.items():
            full = np.full((valid.size, k), np.nan)
            full[valid] = values
            result[f"{label}_{name}"] = full.reshape(*shape, k)
        full_sigma, full_weights = np.full(valid.size, np.nan), np.zeros(valid.size)
        full_sigma[valid], full_weights[valid] = sigma, w
        result[f"sigma_{name}"] = full_sigma.reshape(shape)
        result[f"weights_{name}"] = full_weights.reshape(shape)
        result[f"valid_mask_{name}"] = valid.reshape(shape)
    return result
