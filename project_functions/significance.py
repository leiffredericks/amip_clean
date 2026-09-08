# %% Gridpoint significance, sign agreement, and false-discovery-rate correction
import numpy as np
from scipy import stats

alpha = 0.05


def count_sign_agreement(maps, reference):
    """Count matching nonzero signs: maps (sample, lat, lon), reference (lat, lon)."""
    maps, reference = np.asarray(maps, dtype=float), np.asarray(reference, dtype=float)
    if maps.shape[1:] != reference.shape:
        raise ValueError("reference must match the spatial dimensions of maps.")

    # Missing values never agree. Zero is also not agreement with a nonzero sign,
    # but should still count as a valid sample when calculating agreement fractions.
    valid = np.isfinite(maps) & np.isfinite(reference)
    agrees = ((maps > 0) & (reference > 0)) | ((maps < 0) & (reference < 0))
    return np.sum(valid & agrees, axis=0)


def one_sample_test(maps, alpha=0.05):
    """Two-sided t-test across samples against zero; return pvalue, sig, n_valid."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between zero and one.")

    # Treat both NaNs and infinities as missing, so the test and sample count use
    # exactly the same observations. Samples occupy axis 0, not the time axis.
    maps = np.asarray(maps, dtype=float)
    maps = np.where(np.isfinite(maps), maps, np.nan)
    n_valid = np.sum(np.isfinite(maps), axis=0)

    # Test the mean across independent samples. For multimodel inference, supply
    # one ensemble-mean map per model rather than pooling all realizations.
    # SciPy may still issue warnings for small samples or nearly constant values.
    with np.errstate(invalid="ignore", divide="ignore"):
        result = stats.ttest_1samp(maps, popmean=0., axis=0, nan_policy="omit")

    # Locations with fewer than two valid samples have no usable t-test.
    pvalue = np.where(n_valid >= 2, result.pvalue, np.nan)
    sig = np.isfinite(pvalue) & (pvalue < alpha)
    return pvalue, sig, n_valid


def fdr_bh(pvalue, q=0.05):
    """BH correction across finite input p-values; return pvalue_fdr, sig_fdr."""
    if not 0 < q < 1:
        raise ValueError("q must lie between zero and one.")

    # Flatten only for the multiple-testing calculation. Missing locations stay
    # NaN/False and do not contribute to the number of hypotheses being tested.
    pvalue = np.asarray(pvalue, dtype=float)
    flat_p = pvalue.ravel()
    valid = np.isfinite(flat_p)
    valid_p = flat_p[valid]
    if np.any((valid_p < 0) | (valid_p > 1)):
        raise ValueError("Finite p-values must lie between zero and one.")

    adjusted_flat = np.full(flat_p.shape, np.nan)
    n_tests = valid_p.size
    if n_tests == 0:
        return adjusted_flat.reshape(pvalue.shape), np.zeros(pvalue.shape, dtype=bool)

    # Sort p-values, then scale each by the number of tests divided by its rank.
    # The reverse cumulative minimum enforces the BH step-up adjustment:
    # a smaller raw p-value cannot receive a larger adjusted value.
    order = np.argsort(valid_p)
    ranks = np.arange(1, n_tests + 1)
    adjusted_sorted = valid_p[order] * n_tests / ranks
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0., 1.)

    # Restore the original spatial ordering. Thresholding the adjusted p-values
    # gives the BH rejection mask without calculating a separate rank cutoff.
    adjusted_valid = np.empty(n_tests)
    adjusted_valid[order] = adjusted_sorted
    adjusted_flat[valid] = adjusted_valid
    pvalue_fdr = adjusted_flat.reshape(pvalue.shape)
    sig_fdr = np.isfinite(pvalue_fdr) & (pvalue_fdr <= q)
    return pvalue_fdr, sig_fdr

