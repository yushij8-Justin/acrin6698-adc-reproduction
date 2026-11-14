"""Metadata-only IV18 SOP reference closure (DICOM PS3.3 C.12.4 and 10.3)."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from pydicom.datadict import dictionary_VR

CLASSIC_MR = "1.2.840.10008.5.1.4.1.1.4"
MR_IMAGE = frozenset({CLASSIC_MR, CLASSIC_MR+".1", CLASSIC_MR+".4"})
SEG = "1.2.840.10008.5.1.4.1.1.66.4"
LEGACY_NON_IMAGE = frozenset({
    "1.2.840.10008.3.1.2.3.1",  # Detached Study Management SOP Class
    "1.2.840.10008.3.1.2.5.5",  # Detached Study Management Meta SOP Class
    "1.2.840.10008.3.1.2.3.3",  # Modality Performed Procedure Step
})
INPUT_ROLES = ("ADC", "MASK", "SEG", "TRACE", "SOURCE")
DERIVED_ROLES = INPUT_ROLES[:-1]
REFERENCE_PATH = frozenset({0x00081140, 0x0008114A, 0x0008114B, 0x00081199, 0x00082112})
FRAME_GROUP = 0x52009230
UID_RE = re.compile(r"[0-9]+(?:\.[0-9]+)+\Z")
HASH_RE = re.compile(r"[a-f0-9]{64}\Z")


class SourceStop(RuntimeError):
    """Wrong fixed-bundle identity: stop the batch."""

    def __init__(self, code):
        self.code = code
        self.status = "STOP"
        super().__init__(code)


@dataclass(frozen=True)
class Edge:
    origin_kind: str
    origin_uid: str
    origin_frame: int | None
    path: tuple[int, ...]
    cls: str
    uid: str
    frames: tuple[int, ...] | None
    study_hint: str | None
    series_hint: str | None
    private: bool


def _digest(item):
    return hashlib.sha256(json.dumps(item, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _tag(ds, key):
    return str(getattr(ds, key, "") or "").strip()


def _oid(value):
    return isinstance(value, str) and len(value) <= 64 and UID_RE.fullmatch(value) is not None


def _hash(value):
    return isinstance(value, str) and HASH_RE.fullmatch(value) is not None


def _ints(value):
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        value = [value]
    out = []
    for i in value:
        s = str(i).strip()
        if re.fullmatch(r"[+]?[0-9]+", s) is None or int(s) <= 0:
            raise ValueError
        out.append(int(s))
    if not out:
        raise ValueError
    return tuple(sorted(set(out)))


def _frames_of(ds):
    if _tag(ds, "SOPClassUID") in ((MR_IMAGE - {CLASSIC_MR}) | {SEG}) and not hasattr(ds, "NumberOfFrames"):
        return 0
    try:
        values = _ints(getattr(ds, "NumberOfFrames", "1"))
        return values[0] if len(values) == 1 else 0
    except ValueError:
        return 0


def _unbundle(item):
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
        return tuple(item[0]), item[1]
    return tuple(item), {}


def _walk(ds, kind):
    """Iterate from derived datasets only; do not interpret private tags."""
    uid = _tag(ds, "SOPInstanceUID")

    def recurse(item, path=(), origin_frame=None, study=None, series=None):
        if path:
            study = _tag(item, "StudyInstanceUID") or study
            series = _tag(item, "SeriesInstanceUID") or series
        inst = _tag(item, "ReferencedSOPInstanceUID")
        cls = _tag(item, "ReferencedSOPClassUID")
        if inst or cls or hasattr(item, "ReferencedSOPInstanceUID") or hasattr(item, "ReferencedSOPClassUID"):
            try:
                frames = _ints(getattr(item, "ReferencedFrameNumber", None))
            except ValueError:
                frames = None
            private = any((t >> 16) & 1 for t in path)
            yield Edge(kind, uid, origin_frame, path, cls, inst, frames, study, series, private)
        # Leave pixel data and unrelated scalar values untouched.
        for tag in item.keys():
            raw = item.get_item(tag, keep_deferred=True)
            if raw is None:
                continue
            vr = raw.VR
            if vr is None and not tag.is_private:
                # Implicit VR elements may still be raw and have no VR until
                # accessed. Consult the standard dictionary without loading
                # non-SQ data (notably deferred PixelData).
                try:
                    vr = dictionary_VR(tag)
                except KeyError:
                    continue
            if vr != "SQ":
                continue
            for n, child in enumerate(item[tag].value, 1):
                yield from recurse(child, path+(int(tag),),
                                   n if int(tag) == FRAME_GROUP else origin_frame, study, series)

    yield from recurse(ds)


def _binding(ds, *, patient, study, visit, series):
    return (_tag(ds, "PatientID") == patient and _tag(ds, "StudyInstanceUID") == study
            and _tag(ds, "ClinicalTrialTimePointID") == visit
            and _tag(ds, "SeriesInstanceUID") == series)


def _object_check(ds, reference):
    if _tag(ds, "SOPClassUID") != reference.cls:
        return "REFERENCED_SOP_CLASS_MISMATCH"
    if reference.study_hint and reference.study_hint != _tag(ds, "StudyInstanceUID"):
        return "REFERENCED_STUDY_UID_MISMATCH"
    if reference.series_hint and reference.series_hint != _tag(ds, "SeriesInstanceUID"):
        return "REFERENCED_SERIES_UID_MISMATCH"
    count = _frames_of(ds)
    if not count or reference.frames is None or any(f > count for f in reference.frames):
        return "REFERENCED_FRAME_UNVERIFIED"
    if reference.cls == CLASSIC_MR and reference.frames:
        return "REFERENCED_FRAME_UNVERIFIED"
    return None


def _geom(ds, frame):
    """Known pixel-to-patient geometry for one fixed or proposed source frame."""
    def numbers(value, n):
        vals = tuple(float(v) for v in value)
        if len(vals) != n or any(not math.isfinite(v) for v in vals):
            raise ValueError
        return vals

    def fetch(keyword, macro, item):
        local = getattr(item, macro, None)
        if local is not None:
            if len(local) != 1:
                raise ValueError
            return getattr(local[0], keyword)
        shared = getattr(ds, "SharedFunctionalGroupsSequence", ())
        if shared:
            shared_item = getattr(shared[0], macro, ())
            if len(shared_item) == 1:
                return getattr(shared_item[0], keyword)
            if len(shared_item) > 1:
                raise ValueError
        return getattr(ds, keyword)

    try:
        item = None
        if frame is not None and hasattr(ds, "PerFrameFunctionalGroupsSequence"):
            item = ds.PerFrameFunctionalGroupsSequence[frame - 1]
        orient = numbers(fetch("ImageOrientationPatient", "PlaneOrientationSequence", item), 6)
        spacing = numbers(fetch("PixelSpacing", "PixelMeasuresSequence", item), 2)
        position = numbers(fetch("ImagePositionPatient", "PlanePositionSequence", item), 3)
        frame_uid = _tag(ds, "FrameOfReferenceUID")
        dims = int(ds.Rows), int(ds.Columns)
        if not _oid(frame_uid) or min(spacing) <= 0 or min(dims) <= 0:
            raise ValueError
        return (frame_uid, orient, spacing, position, *dims)
    except (ValueError, AttributeError, IndexError, TypeError, OverflowError):
        return None


def _compatible(left, right):
    return (left is not None and right is not None
            and left[0] == right[0] and left[-2:] == right[-2:]
            and all(abs(a-b) <= 1e-5 for x,y in zip(left[1:3], right[1:3]) for a,b in zip(x,y))
            and math.dist(left[3], right[3]) <= 0.01)


def _geometry_evidence(candidate, origin, edge, adc_geometry):
    target_n = _frames_of(candidate)
    frame_numbers = edge.frames or range(1, target_n+1)
    # A fixed raw source may differ in native geometry. Additional sources
    # need positive geometry evidence plus independently verified provenance.
    targets = [_geom(candidate, f if target_n > 1 else None) for f in frame_numbers]
    if not targets or any(g is None for g in targets):
        return None
    if 0x00082112 in edge.path or 0x00081140 in edge.path:
        if _frames_of(origin) == 1 or edge.origin_frame is not None:
            origin_geom = _geom(origin, edge.origin_frame)
            if any(not _compatible(g, origin_geom) for g in targets):
                return None
    if any(not any(_compatible(g, known) for known in adc_geometry) for g in targets):
        return None
    return targets


def _approval(items, *, ref_key, evidence):
    matches = [i for i in items if isinstance(i, dict) and i.get("ref_key") == ref_key]
    if len(matches) != 1:
        return False
    a = matches[0]
    return (a.get("evidence_sha256") == evidence and a.get("decision") == "APPROVED"
            and isinstance(a.get("reviewer"), str) and bool(a["reviewer"].strip())
            and _hash(a.get("adjudication_record_sha256"))
            and all(a.get(f) is True for f in ("blinded_to_outcomes", "blinded_to_adc_differences",
                                               "independent_of_rule_development", "same_acquisition_confirmed")))


def inspect(objects, *, expected=None, extra_objects=None, approval=None, private_lookup=None):
    """Return a deidentified, metadata-only acquisition provenance decision.

    objects maps ADC/MASK/SEG/TRACE/SOURCE to dataset iterables, optionally
    paired with expected identity. expected can give patient_id, study_uid,
    visit, role, and series_uids. extra_objects maps a candidate series UID to
    {"datasets": iterable, "archive_sha256": str}. Approval may contain reviews
    with ref_key, evidence_sha256, reviewer, adjudication_record_sha256 and
    positive blind/independence/same-acquisition attestations. The caller must
    verify that the adjudication record is authentic and locally retained.
    AI or unsigned records cannot constitute independent approval. Optional
    private_lookup receives exact IDs for local retrieval; never export it.
    """
    result = dict(status="HOLD", reasons=[], unresolved=0, ignored_non_image=0,
                  source_recovered=False, review_items=[], image_reference_count=0,
                  resolved_reference_count=0, provenance_link_count=0,
                  source_series_link_count=0, private_reference_count=0,
                  non_image_management_classes={}, reason_counts={})
    obj = {k:_unbundle(objects.get(k, ())) for k in INPUT_ROLES}
    if any(not value[0] for value in obj.values()):
        result["reasons"] = ["SOURCE_BUNDLE_INCOMPLETE"]
        return result
    expected = expected or {}
    anchor = obj["ADC"][0][0]
    patient = expected.get("patient_id", _tag(anchor, "PatientID"))
    study = expected.get("study_uid", _tag(anchor, "StudyInstanceUID"))
    visit = expected.get("visit", _tag(anchor, "ClinicalTrialTimePointID"))
    if not patient or not _oid(study) or not visit:
        raise SourceStop("SOURCE_FIXED_IDENTITY_MISSING")
    anchor_series = _tag(anchor, "SeriesInstanceUID")
    role = expected.get("role")
    if role is not None and role not in ("FIRST", "REPEAT"):
        raise SourceStop("SOURCE_ACQUISITION_ROLE_INVALID")
    known = {}
    reasons = Counter()
    fixed_series = set()
    seen_uids = set()
    for kind, (dslist, ident) in obj.items():
        series = expected.get("series_uids", {}).get(kind, ident.get("series_uid", _tag(dslist[0], "SeriesInstanceUID")))
        if series in fixed_series or not _oid(series):
            raise SourceStop("SOURCE_FIXED_SERIES_MISMATCH")
        fixed_series.add(series)
        for ds in dslist:
            if not _binding(ds, patient=patient, study=study, visit=visit, series=series):
                raise SourceStop("SOURCE_FIXED_IDENTITY_MISMATCH")
            uid, cls = _tag(ds, "SOPInstanceUID"), _tag(ds, "SOPClassUID")
            if not _oid(uid) or not _oid(cls):
                reasons["SOURCE_FIXED_SOP_UNVERIFIED"] += 1
                continue
            if uid in seen_uids:
                raise SourceStop("SOURCE_DUPLICATE_FIXED_SOP")
            seen_uids.add(uid)
            if cls not in (MR_IMAGE | {SEG} if kind == "SEG" else MR_IMAGE):
                reasons["SOURCE_FIXED_SOP_CLASS_UNSUPPORTED"] += 1
                continue
            known[uid] = (kind, ds)

    candidates = defaultdict(list)
    for series, data in (extra_objects or {}).items():
        if not isinstance(data, dict) or series in fixed_series or not _oid(series) or not _hash(data.get("archive_sha256")):
            continue
        dslist = tuple(data.get("datasets", ()))
        if not dslist or any(not _binding(d, patient=patient, study=study, visit=visit, series=series)
                             or _tag(d, "SOPClassUID") not in MR_IMAGE
                             or not _oid(_tag(d, "SOPInstanceUID"))
                             or _tag(d, "SOPInstanceUID") in seen_uids for d in dslist):
            continue
        for d in dslist:
            candidates[_tag(d, "SOPInstanceUID")].append((series, data["archive_sha256"], d))

    adc_geometry = [_geom(d, f if _frames_of(d) > 1 else None)
                    for d in obj["ADC"][0] for f in range(1, _frames_of(d)+1)] if candidates else []
    review = (approval.get("reviews") or ()) if isinstance(approval, dict) else ()
    management = set()
    private_count = 0
    edges = defaultdict(set)
    for kind in DERIVED_ROLES:
        for ds in obj[kind][0]:
            for e in _walk(ds, kind):
                if e.cls in LEGACY_NON_IMAGE:
                    management.add((e.cls,e.uid))
                elif not _oid(e.cls) or not _oid(e.uid):
                    reasons["SOURCE_REFERENCE_MALFORMED"] += 1
                elif e.private:
                    private_count += 1
                    # Private associations are not affirmative provenance.
                    # Genuine private MR references still need exact closure.
                    if e.cls in MR_IMAGE:
                        edges[e.cls,e.uid,e.frames].add(e)
                elif e.cls not in MR_IMAGE | {SEG}:
                    reasons["SOURCE_REFERENCE_CLASS_UNSUPPORTED"] += 1
                else:
                    edges[e.cls,e.uid,e.frames].add(e)

    items = []
    source_links = set()
    derived_mr_links = set()
    accepted = set()
    resolved = 0
    for (cls, uid, frames), group in sorted(edges.items(), key=lambda x:repr(x[0])):
        ref_key = _digest([anchor_series, role, cls, uid, frames])
        result_item = dict(ref_key=ref_key, sop_class_uid=cls, frames=list(frames or ()),
                           origins=sorted({e.origin_kind for e in group}), occurrences=len(group))
        if private_lookup is not None:
            private_lookup[ref_key] = {"sop_instance_uid":uid, "sop_class_uid":cls,
                 "frames":list(frames or ()), "series_hints":sorted({e.series_hint for e in group if e.series_hint}),
                 "study_hints":sorted({e.study_hint for e in group if e.study_hint})}
        target = known.get(uid)
        if target is not None:
            target_kind, target_ds = target
            problems = {_object_check(target_ds,e) for e in group} - {None}
            if problems:
                reasons.update(problems)
                items.append({**result_item, "status":"HOLD", "reason":sorted(problems)[0]})
                continue
            resolved += 1
            for e in group:
                if (not e.private and e.origin_uid != uid and cls in MR_IMAGE
                        and REFERENCE_PATH.intersection(e.path)):
                    if target_kind == "SOURCE":
                        source_links.add((cls,uid,frames))
                    elif e.origin_kind != target_kind and {e.origin_kind,target_kind} & {"ADC","TRACE"}:
                        derived_mr_links.add((cls,uid,frames))
            continue

        reason = "UNRESOLVED_IMAGE_SOURCE" if cls in MR_IMAGE else "UNRESOLVED_SEG_REFERENCE"
        options = candidates.get(uid, ())
        if len(options) > 1:
            reason = "AMBIGUOUS_ADDITIONAL_SOURCE"
        if len(options) == 1 and cls in MR_IMAGE and role in {"FIRST","REPEAT"}:
            series, archive_sha, extra = options[0]
            geom_proof = []
            for e in sorted(group, key=repr):
                origin = known.get(e.origin_uid)
                if (e.private or origin is None or not REFERENCE_PATH.intersection(e.path)
                        or _object_check(extra,e)):
                    geom_proof = []
                    break
                geometry = _geometry_evidence(extra, origin[1], e, adc_geometry)
                if not geometry:
                    geom_proof = []
                    break
                geom_proof.append([e.origin_kind, e.origin_uid, e.origin_frame, list(e.path), geometry])
            if geom_proof:
                series_hash = _digest(series)
                evidence = _digest([ref_key,series_hash,archive_sha,geom_proof])
                result_item.update(candidate_series_sha256=series_hash,
                                   candidate_archive_sha256=archive_sha, evidence_sha256=evidence,
                                   candidate_geometry_verified=True)
                if _approval(review, ref_key=ref_key, evidence=evidence):
                    accepted.add(ref_key)
                    source_links.add((cls,uid,frames))
                    resolved += 1
                    items.append({**result_item, "status":"APPROVED_ADDITIONAL_SOURCE"})
                    continue
                reason = "SOURCE_INDEPENDENT_REVIEW_REQUIRED"
            else:
                reason = "ADDITIONAL_SOURCE_EVIDENCE_UNVERIFIED"
        reasons[reason] += 1
        items.append({**result_item, "status":"HOLD", "reason":reason})

    # Exact MR linkage among derived objects counts as relationship evidence,
    # but is reported separately from a direct link to the fixed raw series.
    links = source_links | derived_mr_links
    if not links:
        reasons["NO_VERIFIABLE_MR_SOURCE_RELATIONSHIP"] += 1
    classes = Counter(c for c,_ in management)
    result.update(status="HOLD" if reasons else "PASS", reasons=sorted(reasons),
                  unresolved=len(edges)-resolved, ignored_non_image=len(management),
                  source_recovered=bool(accepted) and not reasons, review_items=items,
                  image_reference_count=len(edges), resolved_reference_count=resolved,
                  provenance_link_count=len(links), source_series_link_count=len(source_links),
                  private_reference_count=private_count,
                  non_image_management_classes=dict(sorted(classes.items())),
                  reason_counts=dict(sorted(reasons.items())))
    return result
