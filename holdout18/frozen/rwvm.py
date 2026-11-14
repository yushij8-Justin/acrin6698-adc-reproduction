"""Hold-out B/C reader. This module does not load files or contact external systems.

Only an authenticated, independently signed adjudication can be passed as
`approvals`. The caller verifies the signed file, its author and the acquisition
source gate. This decoder checks the adjudication schema and the metadata
binding again; it does not infer cross-b comparability from a unit string.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re

import numpy as np
from pydicom.datadict import dictionary_VR
from pydicom.tag import Tag
from pydicom.uid import EnhancedMRImageStorage, LegacyConvertedEnhancedMRImageStorage, MRImageStorage

_RWVM = Tag(0x00409096)
_MODALITY_LUT = Tag(0x00283000)
_PRESENTATION_RESCALE = {Tag(0x00281052), Tag(0x00281053)}
_B = (0.0, 100.0, 600.0, 800.0)
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class RWVMGateError(ValueError):
    """An expected hold; systemic identity errors keep the legacy STOP status."""

    def __init__(self, code, status="HOLD"):
        self.code = code
        self.status = status
        super().__init__(code)


def _require(condition, code, status="HOLD"):
    if not condition:
        raise RWVMGateError(code, status)


def _scalar(value, code):
    try:
        x = float(value)
    except (ValueError, TypeError, OverflowError):
        raise RWVMGateError(code) from None
    _require(math.isfinite(x), code)
    return x


def _get(obj, attr, code):
    _require(hasattr(obj, attr), code)
    return getattr(obj, attr)


def _one(ds, keyword, code):
    seq = _get(ds, keyword, code)
    _require(len(seq) == 1, code)
    return seq[0]


def _walk_sequences(ds, root=()):
    """Iterate DICOM metadata locations without dereferencing deferred pixels."""
    for tag in sorted(ds.keys()):
        loc = (*root, int(tag))
        yield tag, loc
        elem = ds.get_item(tag, keep_deferred=True)
        vr = elem.VR
        if vr is None:
            try:
                vr = dictionary_VR(tag)
            except KeyError:
                vr = None
        if vr == "SQ":
            for i, item in enumerate(ds[tag].value):
                yield from _walk_sequences(item, (*loc, i))


def _hash(data):
    return sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _code(ds, issue):
    keys = ("CodeValue", "LongCodeValue", "URNCodeValue")
    values = [(k, str(getattr(ds, k)).strip()) for k in keys if hasattr(ds, k)]
    _require(len(values) == 1 and bool(values[0][1]), issue)
    key, val = values[0]
    scheme = str(getattr(ds, "CodingSchemeDesignator", "")).strip()
    _require(key == "URNCodeValue" or bool(scheme), issue)
    version = str(getattr(ds, "CodingSchemeVersion", "")).strip()
    return (scheme, version, val)


@dataclass(frozen=True)
class LinearMap:
    slope: float
    intercept: float
    first: float
    last: float
    units: tuple
    quantity_digest: str | None


@dataclass(frozen=True)
class Frame:
    dataset: object
    index: int | None
    b: float
    position: tuple
    orientation: tuple
    spacing: tuple
    mapping: LinearMap | None
    mapping_loc: str | None
    padding: tuple | None
    presentation_metadata: tuple


def _padding(ds):
    pairs = (
        ("PixelPaddingValue", "PixelPaddingRangeLimit"),
        ("FloatPixelPaddingValue", "FloatPixelPaddingRangeLimit"),
        ("DoubleFloatPixelPaddingValue", "DoubleFloatPixelPaddingRangeLimit"),
    )
    present = [p for p in pairs if any(hasattr(ds, tag) for tag in p)]
    _require(len(present) < 2, "RWVM_CONFLICTING_PADDING")
    if not present:
        return None
    val, limit = present[0]
    value = _scalar(_get(ds, val, "RWVM_INVALID_PADDING"), "RWVM_INVALID_PADDING")
    bound = _scalar(getattr(ds, limit, value), "RWVM_INVALID_PADDING")
    return (min(value, bound), max(value, bound))


def _linear(item):
    _require(not hasattr(item, "RealWorldValueLUTData"), "RWVM_TABLE_UNSUPPORTED")
    slope = _scalar(_get(item, "RealWorldValueSlope", "RWVM_SLOPE_MISSING"), "RWVM_SLOPE_INVALID")
    intercept = _scalar(_get(item, "RealWorldValueIntercept", "RWVM_INTERCEPT_MISSING"), "RWVM_INTERCEPT_INVALID")
    _require(slope != 0, "RWVM_SLOPE_INVALID")
    ranges = (
        ("RealWorldValueFirstValueMapped", "RealWorldValueLastValueMapped"),
        ("DoubleFloatRealWorldValueFirstValueMapped", "DoubleFloatRealWorldValueLastValueMapped"),
    )
    present = [r for r in ranges if any(hasattr(item, attr) for attr in r)]
    _require(len(present) == 1, "RWVM_RANGE_AMBIGUOUS")
    first, last = tuple(_scalar(_get(item, attr, "RWVM_RANGE_INCOMPLETE"), "RWVM_RANGE_INVALID")
                        for attr in present[0])
    _require(first <= last, "RWVM_RANGE_INVALID")
    units = _code(_one(item, "MeasurementUnitsCodeSequence", "RWVM_UNIT_NOT_UNIQUE"), "RWVM_UNIT_UNCODED")
    # The per-case reviewer must establish physical quantity from actual evidence,
    # even if no Quantity Definition Sequence is supplied. Do not export free text.
    q = getattr(item, "QuantityDefinitionSequence", None)
    digest = None
    if q:
        serial = []
        for content in q:
            # Hash all non-display data elements, including coded concepts,
            # without exporting DICOM quantity-description free text.
            serial.append(content.to_json_dict())
        digest = _hash(serial)
    return LinearMap(slope, intercept, first, last, units, digest)


def _vector(ds, attr, n):
    try:
        vals = tuple(_scalar(v, "RWVM_GEOMETRY_INVALID") for v in _get(ds, attr, "RWVM_GEOMETRY_MISSING"))
    except TypeError:
        raise RWVMGateError("RWVM_GEOMETRY_INVALID") from None
    _require(len(vals) == n, "RWVM_GEOMETRY_INVALID")
    return vals


def _position_groups(frame, shared, keyword):
    present = [d for d in (frame, shared) if d is not None and hasattr(d, keyword)]
    _require(len(present) == 1, "RWVM_GEOMETRY_OR_B_AMBIGUOUS")
    return _one(present[0], keyword, "RWVM_GEOMETRY_OR_B_AMBIGUOUS")


def _mapping_tags(ds):
    entries = list(_walk_sequences(ds))
    return {loc for tag, loc in entries if tag == _RWVM}, tuple(
        (tag, loc) for tag, loc in entries if tag == _MODALITY_LUT or tag in _PRESENTATION_RESCALE
    )


def _presentation_metadata(ds, transforms):
    rows = []
    for tag, location in transforms:
        item = ds
        for i in range(0, len(location)-1, 2):
            item = item[Tag(location[i])].value[location[i+1]]
        value = item[tag].value
        if tag == _MODALITY_LUT:
            # Retain structure/checksum, without exporting free text or LUT arrays.
            value = _hash([x.to_json_dict() for x in value])
        else:
            value = _scalar(value, "RWVM_COPRESENT_RESCALE_INVALID")
        rows.append((location, value))
    return tuple(rows)


def _dataset_frames(ds):
    locations, transforms = _mapping_tags(ds)
    presentation_metadata = _presentation_metadata(ds, transforms)
    padding = _padding(ds)
    uid = str(_get(ds, "SOPClassUID", "RWVM_STORAGE_CLASS_MISSING"))
    top = (int(_RWVM),)
    pfg_tag, sfg_tag = 0x52009230, 0x52009229
    if uid == MRImageStorage:
        _require(not hasattr(ds, "PerFrameFunctionalGroupsSequence") and not hasattr(ds, "SharedFunctionalGroupsSequence"),
                 "RWVM_CLASSIC_MR_FRAME_LAYOUT")
        _require(str(getattr(ds, "NumberOfFrames", 1)) == "1", "RWVM_CLASSIC_MR_FRAME_LAYOUT")
        _require(not locations - {top}, "RWVM_UNSUPPORTED_LOCATION")
        b = _scalar(_get(ds, "DiffusionBValue", "RWVM_B_MISSING"), "RWVM_B_INVALID")
        linear = _linear(_one(ds, "RealWorldValueMappingSequence", "RWVM_ITEMS_AMBIGUOUS")) if locations else None
        yield Frame(ds, None, b, _vector(ds, "ImagePositionPatient", 3),
                    _vector(ds, "ImageOrientationPatient", 6), _vector(ds, "PixelSpacing", 2),
                    linear, "instance" if linear else None, padding, presentation_metadata)
        return
    _require(uid in (EnhancedMRImageStorage, LegacyConvertedEnhancedMRImageStorage), "RWVM_STORAGE_CLASS_UNSUPPORTED")
    per = _get(ds, "PerFrameFunctionalGroupsSequence", "RWVM_PER_FRAME_METADATA_MISSING")
    n = _scalar(_get(ds, "NumberOfFrames", "RWVM_FRAME_COUNT_MISSING"), "RWVM_FRAME_COUNT_INVALID")
    _require(len(per) == n and n >= 1, "RWVM_FRAME_COUNT_INVALID")
    shared = None
    if hasattr(ds, "SharedFunctionalGroupsSequence"):
        shared = _one(ds, "SharedFunctionalGroupsSequence", "RWVM_SHARED_GROUPS_AMBIGUOUS")
    shared_loc = (sfg_tag, 0, int(_RWVM))
    per_locs = {(pfg_tag, i, int(_RWVM)) for i in range(len(per))}
    _require(not locations - ({top, shared_loc} | per_locs), "RWVM_UNSUPPORTED_LOCATION")
    for i, frame in enumerate(per):
        candidates = [(name, obj) for name, obj in (("instance", ds), ("shared", shared), ("frame", frame))
                      if obj is not None and hasattr(obj, "RealWorldValueMappingSequence")]
        _require(len(candidates) <= 1, "RWVM_LOCATION_CONFLICT")
        where, obj = candidates[0] if candidates else (None, None)
        linear = _linear(_one(obj, "RealWorldValueMappingSequence", "RWVM_ITEMS_AMBIGUOUS")) if obj is not None else None
        if linear is None:
            # The original no-transform branch is defined for classic MR only.
            raise RWVMGateError("RWVM_ENHANCED_FRAME_WITHOUT_MAPPING")
        b_group = _position_groups(frame, shared, "MRDiffusionSequence")
        b = _scalar(_get(b_group, "DiffusionBValue", "RWVM_B_MISSING"), "RWVM_B_INVALID")
        position = _position_groups(frame, shared, "PlanePositionSequence")
        orientation = _position_groups(frame, shared, "PlaneOrientationSequence")
        spacing = _position_groups(frame, shared, "PixelMeasuresSequence")
        yield Frame(ds, i, b, _vector(position, "ImagePositionPatient", 3),
                    _vector(orientation, "ImageOrientationPatient", 6), _vector(spacing, "PixelSpacing", 2),
                    linear, where, padding, presentation_metadata)


def _review_frame(frame):
    ds = frame.dataset
    m = frame.mapping
    uid = str(_get(ds, "SOPInstanceUID", "RWVM_SOP_INSTANCE_MISSING"))
    _require(bool(uid), "RWVM_SOP_INSTANCE_MISSING")
    identity_digests = {k: _hash(str(getattr(ds, k, ""))) for k in
                        ("SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID", "FrameOfReferenceUID")}
    fields = dict(
        storage_class=str(ds.SOPClassUID), frame=None if frame.index is None else frame.index + 1,
        b=frame.b, mapping_location=frame.mapping_loc, padding=frame.padding,
        presentation_transform_present=bool(frame.presentation_metadata),
        presentation_transform_metadata=frame.presentation_metadata,
        mapping=None if m is None else {
            "slope": m.slope, "intercept": m.intercept, "first": m.first, "last": m.last,
            "coded_units": m.units, "quantity_definition_sha256": m.quantity_digest,
        },
    )
    binding = {
        "identities_sha256": identity_digests,
        "geometry": (frame.position, frame.orientation, frame.spacing),
        "data_format": tuple(str(getattr(ds, k, "")) for k in
                             ("Rows", "Columns", "PixelRepresentation", "BitsStored", "BitsAllocated")),
        **fields,
    }
    return _hash(binding), fields


def _inspect_frames(dslist):
    _require(bool(dslist), "RWVM_EMPTY_SERIES")
    _require(len({str(getattr(ds, "SOPInstanceUID", "")) for ds in dslist}) == len(dslist), "RWVM_DUPLICATE_INSTANCE")
    frames = [f for ds in dslist for f in _dataset_frames(ds)]
    mapped = [f for f in frames if f.mapping is not None]
    _require(bool(mapped), "RWVM_NO_MAPPING")
    _require({f.b for f in frames} == set(_B), "RWVM_B_VALUES_NOT_FOUR")
    _require(len({f.mapping.units for f in mapped}) == 1, "RWVM_UNITS_DIFFER_ACROSS_FRAMES")
    # Multiple present quantity codes/descriptions cannot silently be considered equivalent.
    digests = {f.mapping.quantity_digest for f in mapped if f.mapping.quantity_digest is not None}
    _require(len(digests) < 2, "RWVM_QUANTITY_DEFINITION_CONFLICT")
    keyed = sorted((_review_frame(f) for f in frames), key=lambda x: x[0])
    metadata_sha256 = _hash([key for key, _ in keyed])
    detail = [{"instance_frame_sha256": key, **fields} for key, fields in keyed]
    return frames, metadata_sha256, detail


def inspect(dslist):
    """Record technical metadata only; does not read any pixel values."""
    dslist = tuple(dslist)
    has_rwvm = any(_mapping_tags(d)[0] for d in dslist)
    if not has_rwvm:
        return {"has_rwvm": False, "eligible_layout": False, "issue": "RWVM_NO_MAPPING",
                "metadata_sha256": None, "frames": []}
    try:
        _, digest, detail = _inspect_frames(dslist)
    except RWVMGateError as exc:
        return {"has_rwvm": True, "eligible_layout": False, "issue": exc.code,
                "metadata_sha256": None, "frames": []}
    return {"has_rwvm": True, "eligible_layout": True, "issue": None,
            "metadata_sha256": digest, "frames": detail}


def _check_approval(approval, digest, frames):
    _require(isinstance(approval, Mapping), "RWVM_INDEPENDENT_REVIEW_MISSING")
    _require(approval.get("decision") == "APPROVED", "RWVM_INDEPENDENT_REVIEW_NOT_APPROVED")
    _require(approval.get("metadata_sha256") == digest, "RWVM_APPROVAL_WRONG_METADATA")
    for field in ("independent_of_rule_development", "outcome_blinded", "adc_difference_blinded",
                  "same_signal_quantity_across_b"):
        _require(approval.get(field) is True, "RWVM_INDEPENDENT_EVIDENCE_INSUFFICIENT")
    for field in ("reviewer", "evidence_ref", "signal_quantity"):
        value = approval.get(field)
        _require(isinstance(value, str) and bool(value.strip()), "RWVM_INDEPENDENT_EVIDENCE_INSUFFICIENT")
    record_hash = approval.get("signed_record_sha256")
    _require(isinstance(record_hash, str) and bool(_SHA.fullmatch(record_hash)), "RWVM_SIGNED_RECORD_HASH_MISSING")
    if any(f.presentation_metadata for f in frames if f.mapping is not None):
        _require(approval.get("presentation_transform_not_required") is True, "RWVM_COPRESENT_TRANSFORM_UNRESOLVED")
    if any(f.mapping is None for f in frames):
        _require(approval.get("mixed_scale_comparable") is True, "RWVM_MIXED_FRAME_SCALES_UNRESOLVED")


def _frame_indices(frames, reference, expected, core):
    """Geometry and patient/series checks match the original TRACE reader."""
    checked = set()
    seen = set()
    indices = []
    for f in frames:
        ds = f.dataset
        if id(ds) not in checked:
            core.identity(ds, expected)
            _require(str(ds.FrameOfReferenceUID) == reference["for_uid"], "FRAME_OF_REFERENCE_MISMATCH", "STOP")
            dims = tuple(_scalar(_get(ds, attr, "RWVM_MATRIX_MISSING"), "RWVM_MATRIX_INVALID")
                         for attr in ("Rows", "Columns"))
            _require(dims == reference["pixels"].shape[1:], "MATRIX_MISMATCH")
            checked.add(id(ds))
        _require(np.max(np.abs(np.asarray(f.orientation) - reference["orientation"])) <= 1e-5, "ORIENTATION_MISMATCH")
        _require(np.max(np.abs(np.asarray(f.spacing) - reference["spacing"])) <= 1e-5, "SPACING_MISMATCH")
        dist = np.linalg.norm(reference["positions"] - np.asarray(f.position), axis=1)
        iz = int(np.argmin(dist))
        _require(float(dist[iz]) <= .01 and np.count_nonzero(dist <= .01) == 1, "POSITION_MISMATCH_OR_AMBIGUOUS")
        bi = _B.index(f.b)
        _require((bi, iz) not in seen, "DUPLICATE_B_SLICE")
        seen.add((bi, iz))
        indices.append((bi, iz))
    _require(len(seen) == len(_B) * len(reference["positions"]), "MISSING_B_SLICE")
    return indices


def _result_reason(fit, mask):
    values = fit["adc"][mask]
    if not np.isfinite(values).all():
        return "NONFINITE_ROI"
    if (values < 0).any():
        return "NEGATIVE_REFERENCE_ADC"
    if not (values > 0).any():
        return "ALL_ZERO_REFERENCE_ADC"
    return None


def apply_and_fit(dslist, reference_volume, expected, mask, approvals=None, *, core=None):
    """Decode only adjudicated TRACE and return B and (when valid) C fits.

    Images are not read until geometry and independent review pass. RWVM
    transforms stored values exactly once. No ROI voxel is dropped.
    """
    if core is None:
        import adc_core as core
    frames, digest, _ = _inspect_frames(tuple(dslist))
    _check_approval(approvals, digest, frames)
    mask = np.asarray(mask)
    shape = reference_volume["pixels"].shape
    _require(mask.dtype == np.dtype(bool) and mask.shape == shape and bool(mask.any()), "RWVM_MASK_INVALID")
    indices = _frame_indices(frames, reference_volume, expected, core)
    mapped, counterfactual = (np.full((len(_B), *shape), np.nan) for _ in range(2))
    decoded = {}
    frames_per_dataset = {}
    for f in frames:
        frames_per_dataset[id(f.dataset)] = frames_per_dataset.get(id(f.dataset), 0) + 1
    for f, (bi, iz) in zip(frames, indices, strict=True):
        ds = f.dataset
        if not mask[iz].any():
            continue
        if id(ds) not in decoded:
            try:
                array = np.asarray(ds.pixel_array)
            except (AttributeError, ValueError, RuntimeError, NotImplementedError):
                raise RWVMGateError("RWVM_PIXEL_DECODE_FAILED") from None
            if f.index is not None:
                if frames_per_dataset[id(ds)] == 1 and array.ndim == 2:
                    array = array[None, :, :]
                _require(array.shape == (frames_per_dataset[id(ds)], *shape[1:]), "RWVM_FRAME_COUNT_OR_MATRIX")
            else:
                _require(array.shape == shape[1:], "MATRIX_MISMATCH")
            decoded[id(ds)] = array
        array = decoded[id(ds)]
        stored = (array if f.index is None else array[f.index])[mask[iz]].astype(np.float64)
        _require(np.isfinite(stored).all(), "RWVM_NONFINITE_STORED")
        if f.padding is not None:
            lo, hi = f.padding
            _require(not np.any((lo <= stored) & (stored <= hi)), "RWVM_PADDING_IN_ROI")
        if f.mapping is not None:
            m = f.mapping
            _require(np.all((m.first <= stored) & (stored <= m.last)), "RWVM_RANGE_NOT_COVERING_ROI")
            slope, intercept = m.slope, m.intercept
        else:
            slope, intercept, _ = core.scaling(ds)
        with np.errstate(over="ignore", invalid="ignore"):
            signal = stored * slope + intercept
        _require(np.isfinite(signal).all(), "RWVM_NONFINITE_SIGNAL")
        mapped[bi, iz][mask[iz]] = signal
        counterfactual[bi, iz][mask[iz]] = stored if f.mapping is not None else signal
    b_fit = core.fit(mapped)
    b_issue = _result_reason(b_fit, mask)
    _require(b_issue is None, "RWVM_B_" + str(b_issue))
    c_fit = core.fit(counterfactual)
    c_issue = _result_reason(c_fit, mask)
    return {"B": b_fit, "C": c_fit if c_issue is None else None, "c_hold_reason": c_issue,
            "metadata_sha256": digest}
