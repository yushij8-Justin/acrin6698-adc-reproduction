"""Reproduce strict-path ADC summaries for the 18-pair same-archive hold-out."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from statistics import stdev
import sys

import numpy as np
import pydicom

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fetch.image_io import InputError, inventory, read_verified_datasets
from holdout18.frozen import adc_core, reference_b, rwvm, source_refs


HERE = Path(__file__).resolve().parent
KINDS = ("ADC", "MASK", "SEG", "TRACE", "SOURCE")
ROLES = ("FIRST", "REPEAT")
ENGINEERING_MARGIN = 0.05e-3
SOLVER_TOLERANCE = 1e-10


class ReplayError(RuntimeError):
    """The replay cannot use these files or match the reported analysis."""


def require(condition, reason):
    if not condition:
        raise ReplayError(reason)


def load_pair_roles():
    provenance = json.loads((HERE / "workflow_provenance.json").read_text())
    module_names = {"adc_core.py", "reference_b.py", "rwvm.py", "source_refs.py"}
    require(set(provenance["frozen_code_sha256"]) == module_names,
            "Unexpected frozen module inventory.")
    for name, digest in provenance["frozen_code_sha256"].items():
        require(sha256((HERE / "frozen" / name).read_bytes()).hexdigest() == digest,
                f"Frozen module changed: {name}.")
    role_bytes = (HERE / "pair_roles.json").read_bytes()
    require(sha256(role_bytes).hexdigest() == provenance["pair_roles_sha256"],
            "The fixed scan and role map changed.")
    role_map = json.loads(role_bytes)
    require(set(role_map) == {"dataset", "visit", "scope", "pairs"}
            and role_map["dataset"] == "ACRIN-6698" and role_map["visit"] == "T0",
            "Expected the ACRIN-6698 baseline role map.")
    pairs = role_map["pairs"]
    require(len(pairs) == 18 and len({p["id"] for p in pairs}) == 18,
            "Expected 18 unique pairs.")
    uids = []
    for pair in pairs:
        require(set(pair) == {"id", "vendor", "acquisitions"}
                and re.fullmatch(r"HO-[0-9]{3}", pair["id"])
                and pair["vendor"] in {"GE", "Philips", "Siemens"},
                "Invalid pair schema.")
        require(tuple(a["role"] for a in pair["acquisitions"]) == ROLES,
                "Each pair must contain first and repeat acquisitions, in order.")
        for acquisition in pair["acquisitions"]:
            require(set(acquisition) == {"role", "combined_source", "series"}
                    and set(acquisition["series"]) == set(KINDS)
                    and type(acquisition["combined_source"]) is bool,
                    "Invalid series role schema.")
            uids.extend(acquisition["series"].values())
    require(len(uids) == len(set(uids)) == 180 and all(
        isinstance(u, str) and len(u) <= 64 and re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", u)
        for u in uids
    ), "Expected 180 distinct DICOM series UIDs.")
    digest = sha256(("\n".join(sorted(uids)) + "\n").encode("ascii")).hexdigest()
    require(digest == provenance["series_uid_set_sha256"], "The fixed series set changed.")
    return pairs, set(uids), provenance


def acquisition_bindings(objects, acquisition):
    """Check deidentified DICOM subject/study locally and never return them in reports."""
    adc_anchor = objects["ADC"][0]
    patient = str(getattr(adc_anchor, "PatientID", ""))
    study = str(getattr(adc_anchor, "StudyInstanceUID", ""))
    require(bool(patient) and bool(study), "Missing baseline subject or study identity.")
    expected = {
        kind: {
            "patient_id": patient,
            "study_uid": study,
            "visit": "T0",
            "series_uid": acquisition["series"][kind],
        }
        for kind in KINDS
    }
    for kind, datasets in objects.items():
        for dataset in datasets:
            adc_core.identity(dataset, expected[kind])
    return expected, (patient, study)


def analyse_acquisition(acquisition, entries, cache):
    objects = {}
    transport_reasons = []
    for kind in KINDS:
        entry = entries[acquisition["series"][kind]]
        if entry["state"] == "HISTORICAL_EXHAUSTED":
            transport_reasons.append(kind + "_SERIES_DOWNLOAD_FAILED")
        else:
            objects[kind] = read_verified_datasets(cache, entry)
    expected, identity = acquisition_bindings(objects, acquisition)
    if transport_reasons:
        return identity, {"reasons": transport_reasons}

    source = source_refs.inspect(objects, expected={
        **expected["ADC"], "role": acquisition["role"], "series_uids": acquisition["series"],
    })
    mapping = rwvm.inspect(objects["TRACE"])
    reasons = list(source["reasons"])
    if mapping["has_rwvm"]:
        reasons.append("TRACE_RWVM_REQUIRES_REVIEW")

    try:
        adc = adc_core.load_volume(objects["ADC"], expected["ADC"], "ADC")
        mask = adc_core.align(adc, adc_core.load_volume(objects["MASK"], expected["MASK"], "MASK")) != 0
        seg = adc_core.align(adc, adc_core.load_volume(objects["SEG"], expected["SEG"], "SEG")) != 0
        adc_core.require(np.array_equal(mask, seg), "MASK_SEG_SUPPORT_MISMATCH")
        adc_core.require(bool(mask.any()), "EMPTY_ROI")
        public = adc["physical"][mask]
        adc_core.require(np.isfinite(public).all(), "NONFINITE_PUBLIC_ADC")
        adc_core.require((public >= 0).all(), "NEGATIVE_PUBLIC_ADC")
        adc_core.require((public > 0).any(), "ALL_ZERO_PUBLIC_ADC")
        if all(hasattr(dataset, "DiffusionBValue") for dataset in objects["TRACE"]):
            b_values = sorted({float(dataset.DiffusionBValue) for dataset in objects["TRACE"]})
            adc_core.require(b_values == [0, 100, 600, 800], "B_VALUES")
        if not mapping["has_rwvm"]:
            trace = adc_core.trace_signal(objects["TRACE"], adc, expected["TRACE"])
        if reasons:
            return identity, {"reasons": reasons}

        fit = adc_core.fit(trace["signal"])
        independent = reference_b.fit(reference_b.from_datasets(objects["TRACE"])["signal"])
        require(np.array_equal(fit["reason"][mask], independent["reason"][mask]),
                "The two TRACE solvers produced different reason masks.")
        solver_error = float(np.max(np.abs(fit["adc"][mask] - independent["adc"][mask])))
        require(math.isfinite(solver_error) and solver_error <= SOLVER_TOLERANCE,
                "The two TRACE solvers disagreed beyond the frozen tolerance.")
        result = adc_core.roi_readout(adc["physical"], fit["adc"], mask)
        return identity, {
            "reasons": [],
            "public_mean": result["P_mean"],
            "recalculated_mean": result["U_mean"],
            "solver_max_abs_mm2_s": solver_error,
        }
    except adc_core.GateError as exc:
        if exc.status == "STOP":
            raise
        return identity, {"reasons": [*reasons, exc.code]}
    except (pydicom.errors.InvalidDicomError, NotImplementedError):
        return identity, {"reasons": [*reasons, "LOCAL_DICOM_READ_OR_DECODER_UNSUPPORTED"]}


def within_subject_cv(log_ratios):
    return 100 * math.sqrt(math.expm1(stdev(log_ratios) ** 2 / 2))


def analyse_pairs(pairs, entries, cache):
    reasons = Counter()
    vendor_attempted = Counter()
    vendor_accepted = Counter()
    patients = set()
    complete_transport_pairs = complete_transport_acquisitions = accepted_acquisitions = 0
    solver_errors, differences, public_ratios, recalculated_ratios = [], [], [], []
    transfer_holds = rwvm_overlap = 0
    for pair in pairs:
        vendor_attempted[pair["vendor"]] += 1
        evaluated = [analyse_acquisition(a, entries, cache) for a in pair["acquisitions"]]
        first_identity, first = evaluated[0]
        repeat_identity, repeat = evaluated[1]
        require(first_identity == repeat_identity, "The two scans do not share subject and study identity.")
        require(first_identity[0] not in patients, "The same subject occurs in multiple pairs.")
        patients.add(first_identity[0])

        pair_complete = True
        for row in (first, repeat):
            unique_reasons = set(row["reasons"])
            reasons.update(unique_reasons)
            if any(reason.endswith("_SERIES_DOWNLOAD_FAILED") for reason in unique_reasons):
                transfer_holds += 1
                pair_complete = False
            else:
                complete_transport_acquisitions += 1
            if {"UNRESOLVED_IMAGE_SOURCE", "TRACE_RWVM_REQUIRES_REVIEW"} <= unique_reasons:
                rwvm_overlap += 1
            if not unique_reasons:
                accepted_acquisitions += 1
                solver_errors.append(row["solver_max_abs_mm2_s"])
        if pair_complete:
            complete_transport_pairs += 1
        if first["reasons"] or repeat["reasons"]:
            continue

        vendor_accepted[pair["vendor"]] += 1
        public = [row["public_mean"] for row in (first, repeat)]
        recalculated = [row["recalculated_mean"] for row in (first, repeat)]
        differences.append(math.fsum(p - u for p, u in zip(public, recalculated)) / 2)
        public_ratios.append(math.log(public[1] / public[0]))
        recalculated_ratios.append(math.log(recalculated[1] / recalculated[0]))

    require(len(differences) >= 2, "Too few strict complete pairs for the reported descriptive summaries.")
    return {
        "dataset": "ACRIN-6698",
        "scope": "same-archive hold-out",
        "analysis": "strict only, descriptive",
        "attempted_pairs": len(pairs),
        "attempted_acquisitions": 2 * len(pairs),
        "specified_series": len(entries),
        "checksum_verified_series": sum(x["state"] == "AVAILABLE" for x in entries.values()),
        "transport_attempt_limit": sum(x["state"] == "HISTORICAL_EXHAUSTED" for x in entries.values()),
        "checksum_verified_DICOM_files": sum(x["instances"] for x in entries.values() if x["state"] == "AVAILABLE"),
        "complete_transport_pairs": complete_transport_pairs,
        "complete_transport_acquisitions": complete_transport_acquisitions,
        "strict_A_pass_pairs": len(differences),
        "strict_A_pass_acquisitions": accepted_acquisitions,
        "strict_A_transport_holds": transfer_holds,
        "strict_A_unresolved_image_references": reasons["UNRESOLVED_IMAGE_SOURCE"],
        "strict_A_RWVM_overlap": rwvm_overlap,
        "strict_reason_counts": dict(sorted(reasons.items())),
        "attempted_pairs_by_vendor": dict(sorted(vendor_attempted.items())),
        "strict_pairs_by_vendor": dict(sorted(vendor_accepted.items())),
        "strict_A_patient_weighted_mean_released_minus_recalculated":
            1000 * math.fsum(differences) / len(differences),
        "strict_A_released_ADC_wCV": within_subject_cv(public_ratios),
        "strict_A_recalculated_ADC_wCV": within_subject_cv(recalculated_ratios),
        "B_newly_recovered_pairs": None,
        "B_C_mapped_vs_stored_absolute_difference": None,
        "B_C_prespecified_engineering_margin": ENGINEERING_MARGIN * 1000,
        "B_recovery_status": "NOT_ADJUDICATED",
        "B_C_mapping_status": "NOT_ESTIMABLE",
        "independent_solver_max_abs_mm2_s": max(solver_errors),
        "clinical_outcomes_accessed": False,
    }


def verify_report(summary, provenance):
    path = HERE / "published_strict_results.tsv"
    require(sha256(path.read_bytes()).hexdigest() == provenance["published_strict_results_sha256"],
            "The archived reference summary changed.")
    with path.open(newline="") as handle:
        witness = {r["metric"]: r["count_or_value"] for r in csv.DictReader(handle, delimiter="\t")}
    for name, expected in witness.items():
        result = summary[name]
        if not expected:
            valid = result is None
        elif "." in expected:
            decimals = len(expected.partition(".")[2])
            valid = f"{result:.{decimals}f}" == expected
        else:
            valid = result == int(expected)
        require(valid, f"Published strict-path value did not reproduce: {name}.")
    require(summary["strict_pairs_by_vendor"] == {"GE": summary["strict_A_pass_pairs"]},
            "The qualifying pairs no longer match the reported manufacturer.")
    summary["published_precision_check"] = "PASS"
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path, help="Directory with verified UID-named series ZIPs.")
    parser.add_argument("--output", required=True, type=Path, help="New local directory to receive summary.json.")
    args = parser.parse_args()
    require(not args.output.exists(), "Choose a new output directory.")

    pairs, uids, provenance = load_pair_roles()
    entries = inventory("holdout18")
    require(set(entries) == uids, "The inventory and role map use different series.")
    require(Counter(e["state"] for e in entries.values()) ==
            {"AVAILABLE": 170, "HISTORICAL_EXHAUSTED": 10}, "Unexpected terminal transport inventory.")
    require(all(entries[a["series"]["ADC"]]["state"] == "AVAILABLE"
                for p in pairs for a in p["acquisitions"]), "A fixed ADC identity anchor is unavailable.")

    summary = verify_report(analyse_pairs(pairs, entries, args.cache), provenance)
    args.output.mkdir(mode=0o700, parents=True)
    with (args.output / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Hold-out replay matched the reported values for {summary['strict_A_pass_pairs']}/"
          f"{summary['attempted_pairs']} complete pairs. See {args.output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ReplayError, InputError, adc_core.GateError, source_refs.SourceStop) as exc:
        print(f"Hold-out replay stopped: {exc}", file=sys.stderr)
        sys.exit(1)
