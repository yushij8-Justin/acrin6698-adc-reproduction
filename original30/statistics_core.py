"""Frozen whole-patient, within-site resampling statistics. No file I/O."""
from __future__ import annotations

import hashlib
import math
from collections import Counter

import numpy as np


SEED = 20260923
REPLICATES = 10000
FIELDS = ("P_first", "P_repeat", "U_first", "U_repeat")


def bootstrap_indices(sites, replicates=REPLICATES, seed=SEED):
    """Return (replicates, patients) whole-patient indices into input sites.

    Generate each site's complete matrix, in lexicographic site order, using
    one fresh PCG64 instance; concatenate site matrices across columns.
    """
    if isinstance(replicates, bool) or int(replicates) != replicates or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    labels = [str(site) for site in sites]
    rng = np.random.Generator(np.random.PCG64(seed))
    blocks = []
    for site in sorted(set(labels)):
        members = np.asarray([i for i, label in enumerate(labels) if label == site], dtype=np.int64)
        draws = rng.integers(0, len(members), size=(int(replicates), len(members)))
        blocks.append(members[draws])
    if not blocks:
        return np.empty((int(replicates), 0), dtype=np.int64)
    return np.concatenate(blocks, axis=1).astype(np.int64, copy=False)


def _finite(value):
    try:
        return value is not None and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _number(value):
    """Always produce strict-JSON-safe numerical output."""
    return float(value) if _finite(value) else None


def _ci(values):
    if not np.isfinite(values).all():
        return None
    return [_number(x) for x in np.quantile(values, [0.025, 0.975], method="linear")]


def _descriptive(values):
    values = np.asarray(values, dtype=np.float64)
    return {"n": int(len(values)), "mean": _number(np.mean(values)) if len(values) else None,
            "median": _number(np.median(values)) if len(values) else None}


def _wcv(logratios, axis=None):
    with np.errstate(over="ignore", invalid="ignore"):
        variance = np.var(logratios, axis=axis, ddof=1)
        return 100.0 * np.sqrt(np.expm1(variance / 2.0))


def summarize(records):
    """Summarize explicit in-memory ROI means (mm2/s), with missingness audit.

    Finite complete patients define the primary cohort. Nonpositive ROI means
    do not become missing or change that cohort, but make logratio statistics
    unavailable for the full cohort. No imputation or positive-only replacement.
    """
    records = list(records)
    ordered = []
    seen = set()
    invalid_fields = Counter({field: 0 for field in FIELDS})
    missing_metadata = 0
    excluded = []
    site_input = Counter()
    for record in records:
        patient_id, site = record.get("patient_id"), record.get("site")
        metadata_valid = patient_id is not None and site is not None and str(patient_id) != "" and str(site) != ""
        if metadata_valid:
            patient_id, site = str(patient_id), str(site)
            if patient_id in seen:
                raise ValueError("duplicate patient_id: whole-patient records must be unique")
            seen.add(patient_id)
            site_input[site] += 1
        else:
            missing_metadata += 1
        bad_fields = [field for field in FIELDS if not _finite(record.get(field))]
        invalid_fields.update(bad_fields)
        if not metadata_valid or bad_fields:
            excluded.append({"patient_id": str(patient_id) if patient_id is not None else None,
                             "site": str(site) if site is not None else None,
                             "invalid_fields": bad_fields, "invalid_identity": not metadata_valid})
        else:
            ordered.append((site, patient_id, [float(record[field]) for field in FIELDS]))
    ordered.sort(key=lambda row: (row[0], row[1]))
    sites = [row[0] for row in ordered]
    ids = [row[1] for row in ordered]
    values = np.asarray([row[2] for row in ordered], dtype=np.float64).reshape(-1, 4)
    n = len(values)
    indices = bootstrap_indices(sites)
    site_counts = dict(sorted(Counter(sites).items()))
    pf, pr, uf, ur = values.T
    with np.errstate(over="ignore", invalid="ignore"):
        first = 1000.0 * (pf - uf)
        repeat = 1000.0 * (pr - ur)
        d = 1000.0 * ((pf - uf) + (pr - ur)) / 2.0
    theta = _number(np.mean(d)) if n else None
    interval_allowed = n >= 24
    interval = _ci(np.mean(d[indices], axis=1)) if interval_allowed else None
    halfwidth = (interval[1] - interval[0]) / 2.0 if interval is not None else None
    primary = {"n": n, "theta": theta, "ci": interval, "halfwidth": halfwidth,
               "unit": "10^-3 mm2/s", "precision_target": 0.05,
               "precision_target_met": halfwidth <= 0.05 if halfwidth is not None else None,
               "theta_minus_lower": theta - interval[0] if interval is not None else None,
               "upper_minus_theta": interval[1] - theta if interval is not None else None,
               "interval_status": "available" if interval is not None else ("n_below_24" if not interval_allowed else "nonfinite_statistic"),
               "interval_excludes_zero": bool(interval[0] > 0 or interval[1] < 0) if interval is not None else None,
               "patient_ids": ids, "patient_d": [_number(x) for x in d]}
    secondary = {"first_difference": _descriptive(first), "repeat_difference": _descriptive(repeat),
                 "repeat_minus_first_difference": _descriptive(repeat - first),
                 "patient_mean_absolute_difference": _descriptive((np.abs(first) + np.abs(repeat)) / 2.0),
                 "unit": "10^-3 mm2/s"}
    log_eligible = np.all(values > 0, axis=1)
    repeatability = {"n": n, "positive_complete_n": int(log_eligible.sum()),
                     "nonpositive_complete_n": int((~log_eligible).sum()), "P": None, "U": None,
                     "wcv_difference_P_minus_U": None, "wcv_difference_ci": None,
                     "status": "available" if n >= 2 and log_eligible.all() else ("fewer_than_two" if n < 2 else "nonpositive_roi_mean")}
    if n >= 2 and log_eligible.all():
        # Difference of logs avoids overflowing a finite repeat/first ratio.
        rp, ru = np.log(pr) - np.log(pf), np.log(ur) - np.log(uf)
        bootstrap_wcv = {}
        for label, ratios in (("P", rp), ("U", ru)):
            mean, sd = float(np.mean(ratios)), float(np.std(ratios, ddof=1))
            wcv = _number(_wcv(ratios))
            item = {"logratio_mean": mean, "logratio_sd": sd,
                    "logratio_loa": [mean - 1.96 * sd, mean + 1.96 * sd],
                    "wcv_percent": wcv, "wcv_ci": None}
            if interval_allowed:
                bootstrap_wcv[label] = _wcv(ratios[indices], axis=1)
                item["wcv_ci"] = _ci(bootstrap_wcv[label])
            repeatability[label] = item
        wp, wu = repeatability["P"]["wcv_percent"], repeatability["U"]["wcv_percent"]
        repeatability["wcv_difference_P_minus_U"] = wp - wu if wp is not None and wu is not None else None
        if interval_allowed:
            repeatability["wcv_difference_ci"] = _ci(bootstrap_wcv["P"] - bootstrap_wcv["U"])
    secondary["repeatability"] = repeatability
    # Available single-acquisition pairs are descriptive only, even if their
    # patient's other acquisition is unavailable.
    single = {}
    for role, pfield, ufield in (("first", "P_first", "U_first"), ("repeat", "P_repeat", "U_repeat")):
        diffs = [1000.0 * (float(r[pfield]) - float(r[ufield])) for r in records
                 if _finite(r.get(pfield)) and _finite(r.get(ufield))]
        single[role] = _descriptive(diffs)
    secondary["available_single_acquisitions"] = single
    return {"primary": primary, "secondary": secondary,
            "counters": {"input_records": len(records), "complete_primary": n,
                         "excluded_primary": len(records) - n, "invalid_identity": missing_metadata,
                         "invalid_fields": dict(invalid_fields), "excluded_records": excluded,
                         "site_input_counts": dict(sorted(site_input.items()))},
            "bootstrap": {"seed": SEED, "rng": "PCG64", "replicates": REPLICATES,
                          "index_shape": list(indices.shape),
                          "index_sha256": hashlib.sha256(indices.astype("<i8", copy=False).tobytes(order="C")).hexdigest(),
                          "index_encoding": "little-endian int64 C-order",
                          "site_counts": site_counts,
                          "singleton_sites": [site for site, count in site_counts.items() if count == 1],
                          "patient_ids": ids, "patient_sites": sites,
                          "quantile_method": "linear", "all_intervals_share_indices": True}}
