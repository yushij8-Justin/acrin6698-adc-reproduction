"""Reproduce the original 30-pair ADC analysis from checksum-verified DICOM."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import pydicom

import adc_core
import reference_b
from statistics_core import summarize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fetch.image_io import inventory, read_verified_series


ROLES = ("FIRST", "REPEAT")
KINDS = ("ADC", "MASK", "SEG", "TRACE", "SOURCE")
VENDORS = {
    "GE MEDICAL SYSTEMS": "GE",
    "Philips Medical Systems": "Philips",
    "SIEMENS": "Siemens",
}
MR_SOP_CLASSES = {
    "1.2.840.10008.5.1.4.1.1.4",
    "1.2.840.10008.5.1.4.1.1.4.1",
    "1.2.840.10008.5.1.4.1.1.4.4",
}
DETACHED_STUDY_SOP_CLASS = "1.2.840.10008.3.1.2.3.1"
MPPS_SOP_CLASS = "1.2.840.10008.3.1.2.3.3"
GLOBAL_STOPS = {
    "UNKNOWN_ADC_UNIT",
    "INVALID_PUBLIC_SCALING",
    "SLICE_SCALING_VARIES",
    "EXTRA_RWVM_OR_LUT",
}
POINT_FIELDS = (
    "participant",
    "first_difference_1e-3_mm2_s",
    "repeat_difference_1e-3_mm2_s",
    "mean_difference_1e-3_mm2_s",
    "public_log_repeat_first",
    "reference_log_repeat_first",
)


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def load_roster(expected):
    rows = json.loads((HERE / "roster.json").read_text(encoding="utf-8"))
    require(len(rows) == 30, "The original roster must contain 30 participants.")
    subjects, uids = set(), set()
    for row in rows:
        require(set(row) == {"participant", "manufacturer", "acquisitions"},
                "Unexpected original roster fields.")
        subject = row["participant"]
        require(subject not in subjects and row["manufacturer"] in VENDORS.values(),
                "Original roster has a duplicate label or an unrecognized vendor.")
        subjects.add(subject)
        require(set(row["acquisitions"]) == set(ROLES), "A paired acquisition role is missing.")
        for acquisition in row["acquisitions"].values():
            require(set(acquisition) == {"series", "source_is_merged_msmb"},
                    "Unexpected acquisition fields.")
            series = acquisition["series"]
            require(set(series) == set(KINDS), "An original acquisition needs five named series.")
            require(type(acquisition["source_is_merged_msmb"]) is bool,
                    "The source category must be explicitly stated.")
            for uid in series.values():
                require(uid in expected and uid not in uids,
                        "A series is missing from the integrity inventory or assigned twice.")
                uids.add(uid)
    require(uids == set(expected) and len(uids) == 300,
            "Roster series and the frozen original integrity inventory differ.")
    return rows


def read_inputs(cache, series, expected, declared_vendor):
    bundles = {}
    patient, study, sop_uids = None, None, set()
    for kind in KINDS:
        uid = series[kind]
        objects = [pydicom.dcmread(io.BytesIO(item)) for item in read_verified_series(cache, expected[uid])]
        require(objects, "A required image series is empty.")
        if patient is None:
            patient = str(getattr(objects[0], "PatientID", ""))
            study = str(getattr(objects[0], "StudyInstanceUID", ""))
            require(patient and study, "Required DICOM identity fields are missing.")
        identity = {"patient_id": patient, "study_uid": study, "series_uid": uid, "visit": "T0"}
        for instance in objects:
            adc_core.identity(instance, identity)
            sop = str(getattr(instance, "SOPInstanceUID", ""))
            require(sop and sop not in sop_uids, "An acquisition has a duplicate or absent SOP UID.")
            sop_uids.add(sop)
        bundles[kind] = objects, identity
    require(all(VENDORS.get(str(getattr(ds, "Manufacturer", ""))) == declared_vendor
                for ds in bundles["ADC"][0]), "Published ADC vendor conflicts with the roster.")
    return bundles, patient, study, sop_uids


def reference_classes(derived, available):
    """Classify the objects missing inside the fixed series set, without reporting UIDs."""
    missing_classes = set()
    for dataset in derived:
        for element in dataset.iterall():
            if element.VR != "SQ":
                continue
            for item in element.value:
                uid = str(getattr(item, "ReferencedSOPInstanceUID", ""))
                sop_class = str(getattr(item, "ReferencedSOPClassUID", ""))
                if uid and uid not in available:
                    missing_classes.add(sop_class)
    return {
        "missing_mr": bool(missing_classes & MR_SOP_CLASSES),
        "missing_mpps": MPPS_SOP_CLASS in missing_classes,
        "missing_detached_study": DETACHED_STUDY_SOP_CLASS in missing_classes,
    }


def fit_and_readout(label, role, bundles, available, merged_source):
    result = {"participant": label, "role": role, "status": "HOLD"}
    adc, adc_identity = bundles["ADC"]
    mask, mask_identity = bundles["MASK"]
    seg, seg_identity = bundles["SEG"]
    trace, trace_identity = bundles["TRACE"]
    try:
        volume = adc_core.load_volume(adc, adc_identity, "ADC")
        mask = adc_core.align(volume, adc_core.load_volume(mask, mask_identity, "MASK")) != 0
        seg = adc_core.align(volume, adc_core.load_volume(seg, seg_identity, "SEG")) != 0
        result["mask_seg_mismatches"] = int(np.count_nonzero(seg != mask))
        adc_core.require(result["mask_seg_mismatches"] == 0, "MASK_SEG_SUPPORT_MISMATCH")
        trace_signal = adc_core.trace_signal(trace, volume, trace_identity)
        derived = [obj for kind in KINDS[:-1] for obj in bundles[kind][0]]
        try:
            adc_core.audit_references(derived, available)
        except adc_core.GateError:
            result["reference_evidence"] = reference_classes(derived, available)
            raise
        fit = adc_core.fit(trace_signal["signal"])
        result["readout"] = adc_core.roi_readout(volume["physical"], fit["adc"], mask)

        independent = reference_b.from_datasets(trace)
        require(np.allclose(independent["positions"], volume["positions"], rtol=0, atol=.01),
                "Independent TRACE geometry cannot be reconciled with the released ADC.")
        check = reference_b.fit(independent["signal"][:, mask])
        require(np.array_equal(check["reason"], fit["reason"][mask]),
                "Independent TRACE fit returned different voxel dispositions.")
        absolute_error = np.abs(check["adc"] - fit["adc"][mask])
        maximum_error = float(np.max(absolute_error))
        require(maximum_error <= 1e-10, "Independent TRACE solver differs beyond the frozen tolerance.")
        result["solver_max_abs_adc_error_mm2_s"] = maximum_error
        result["threshold_secondary_applicable"] = False
        if not merged_source:
            source, source_identity = bundles["SOURCE"]
            try:
                original = adc_core.trace_signal(source, volume, source_identity)
                result["threshold_secondary_applicable"] = bool(
                    original["all_public_rescale_explicit"]
                    and len(original["scales"]) == 1
                    and original["scales"][0][1] == 0
                    and np.array_equal(original["raw"], trace_signal["raw"])
                )
            except adc_core.GateError as error:
                if error.status == "STOP":
                    raise
            except AttributeError:
                # Absent metadata only limits this prespecified secondary calculation.
                pass
        result["status"] = "PASS"
    except adc_core.GateError as error:
        if error.status == "STOP" or error.code in GLOBAL_STOPS:
            raise ValueError(f"{label} {role}: systemic stop ({error.code})") from None
        result["reason"] = error.code
    return result


def fraction_distribution(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def table_results(roster, rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["participant"], {})[row["role"]] = row
    numeric_subjects, figure = [], []
    table1 = {vendor: Counter() for vendor in VENDORS.values()}
    for subject in roster:
        label = subject["participant"]
        pair = grouped[label]
        passed = sum(pair[role]["status"] == "PASS" for role in ROLES)
        counts = table1[subject["manufacturer"]]
        counts.update({
            "attempted_pairs": 1,
            "complete_pairs": int(passed == 2),
            "held_pairs": int(passed < 2),
            "pass_acquisitions": passed,
            "held_acquisitions": 2 - passed,
        })
        # With the fixed 17 complete pairs the historical within-site interval is unavailable.
        # A constant grouping avoids redistributing acquisition-site relationships.
        numeric = {"patient_id": label, "site": "not_used_below_minimum_sample"}
        for role in ROLES:
            readout = pair[role].get("readout", {})
            numeric[f"P_{role.lower()}"] = readout.get("P_mean")
            numeric[f"U_{role.lower()}"] = readout.get("U_mean")
        numeric_subjects.append(numeric)
        if passed == 2:
            p0, p1 = numeric["P_first"], numeric["P_repeat"]
            u0, u1 = numeric["U_first"], numeric["U_repeat"]
            first = 1000 * (p0 - u0)
            repeat = 1000 * (p1 - u1)
            figure.append((label, first, repeat, (first + repeat) / 2,
                           math.log(p1 / p0), math.log(u1 / u0)))
    stats = summarize(numeric_subjects)
    require(stats["primary"]["n"] < 24, "Unexpected cohort disposition requires a new review.")
    good = [row["readout"] for row in rows if row["status"] == "PASS"]
    require(good and len(good) >= len(figure) * 2, "No valid original-cohort readouts.")
    secondary = stats["secondary"]
    repeatability = secondary["repeatability"]
    table2 = {
        "patient_weighted_mean_difference": stats["primary"]["theta"],
        "first_mean_difference": secondary["first_difference"]["mean"],
        "repeat_mean_difference": secondary["repeat_difference"]["mean"],
        "mean_change_repeat_minus_first": secondary["repeat_minus_first_difference"]["mean"],
        "patient_absolute_difference_mean": secondary["patient_mean_absolute_difference"]["mean"],
        "patient_absolute_difference_median": secondary["patient_mean_absolute_difference"]["median"],
        "public_wcv_percent": repeatability["P"]["wcv_percent"],
        "reference_wcv_percent": repeatability["U"]["wcv_percent"],
        "public_minus_reference_wcv_percentage_points": repeatability["wcv_difference_P_minus_U"],
        "zero_support_mismatch": sum(r["zero_support_mismatch"] for r in good),
    }
    additional = {
        "public_zero_fraction": fraction_distribution([x["public_zero"] / x["roi_voxels"] for x in good]),
        "reference_zero_fraction": fraction_distribution([x["reference_zero"] / x["roi_voxels"] for x in good]),
        "common_positive_difference": statistics.mean(
            1000 * (x["common_positive_P_mean"] - x["common_positive_U_mean"])
            for x in good if x["common_positive_n"]
        ),
        "public_logratio_mean": repeatability["P"]["logratio_mean"],
        "public_logratio_descriptive_limits": repeatability["P"]["logratio_loa"],
        "reference_logratio_mean": repeatability["U"]["logratio_mean"],
        "reference_logratio_descriptive_limits": repeatability["U"]["logratio_loa"],
        "threshold_applicable_acquisitions": sum(bool(x.get("threshold_secondary_applicable")) for x in rows),
    }
    summary = {
        "cohort": "Original ACRIN-6698 baseline pairs",
        "attempted_pairs": len(roster),
        "attempted_acquisitions": len(rows),
        "complete_pairs": len(figure),
        "pass_acquisitions": len(good),
        "hold_reasons": dict(sorted(Counter(x["reason"] for x in rows if x["status"] == "HOLD").items())),
        "holds_with_unresolved_mr_references": sum(
            bool(x.get("reference_evidence", {}).get("missing_mr")) for x in rows
        ),
        "holds_with_unresolved_mr_and_mpps_references": sum(
            bool(x.get("reference_evidence", {}).get("missing_mr"))
            and bool(x["reference_evidence"].get("missing_mpps")) for x in rows
        ),
        "holds_with_unresolved_mr_and_ignored_detached_study_references": sum(
            bool(x.get("reference_evidence", {}).get("missing_mr"))
            and bool(x["reference_evidence"].get("missing_detached_study")) for x in rows
        ),
        "mask_seg_support_matches": sum(x.get("mask_seg_mismatches") == 0 for x in rows),
        "maximum_independent_solver_error_mm2_s": max(
            x["solver_max_abs_adc_error_mm2_s"] for x in rows if x["status"] == "PASS"
        ),
        "primary_inference": stats["primary"]["interval_status"],
        "table_1": {key: dict(value) for key, value in table1.items()},
        "table_2": table2,
        "additional_descriptive_results": additional,
    }
    return summary, sorted(figure)


def compare(summary):
    reference = json.loads((HERE / "expected_summary.json").read_text(encoding="utf-8"))
    for key, value in reference["dispositions"].items():
        require(summary[key] == value, f"The reproduced disposition for {key} differs from the archived report.")
    for metric, expected in reference["table_2"].items():
        require(math.isclose(summary["table_2"][metric], expected, rel_tol=0, abs_tol=1e-12),
                f"The reproduced Table 2 value for {metric} differs from the archived report.")
    for metric, expected in reference["additional_descriptive_results"].items():
        actual = summary["additional_descriptive_results"][metric]
        if isinstance(expected, dict):
            matched = all(math.isclose(actual[k], v, rel_tol=0, abs_tol=1e-12) for k, v in expected.items())
        elif isinstance(expected, list):
            matched = len(actual) == len(expected) and all(
                math.isclose(a, b, rel_tol=0, abs_tol=1e-12) for a, b in zip(actual, expected)
            )
        else:
            matched = math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
        require(matched, f"The reproduced {metric} differs from the archived report.")


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(cache, output):
    require(not output.exists(), "Choose a new output directory.")
    expected = inventory("original30")
    roster = load_roster(expected)
    identities, rows = set(), []
    for subject in roster:
        label = subject["participant"]
        previous = None
        for role in ROLES:
            acquisition = subject["acquisitions"][role]
            try:
                bundles, patient, study, available = read_inputs(
                    cache, acquisition["series"], expected, subject["manufacturer"]
                )
            except adc_core.GateError as error:
                raise ValueError(f"{label} {role}: DICOM identity check failed ({error.code})") from None
            if previous is None:
                require(patient not in identities, "The original cohort assigns one DICOM patient twice.")
                identities.add(patient)
                previous = patient, study
            else:
                require((patient, study) == previous, "Paired original acquisitions have different DICOM identities.")
            rows.append(fit_and_readout(label, role, bundles, available, acquisition["source_is_merged_msmb"]))
    summary, figure = table_results(roster, rows)
    summary["checksum_verified_series"] = len(expected)
    summary["checksum_verified_instances"] = sum(entry["instances"] for entry in expected.values())
    compare(summary)
    output.mkdir(parents=True, mode=0o700)
    write_json(output / "summary.json", summary)
    write_json(output / "technical_readouts.local.json", rows)
    with (output / "figure_2_points.local.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(POINT_FIELDS)
        writer.writerows(figure)
    print(f"Original cohort verified: {summary['complete_pairs']} of {summary['attempted_pairs']} complete pairs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.cache, args.output)


if __name__ == "__main__":
    main()
