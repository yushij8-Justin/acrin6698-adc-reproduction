"""Independent four-b TRACE reference, with no production-A imports."""
from __future__ import annotations

import numpy as np


B_VALUES = (0.0, 100.0, 600.0, 800.0)


def fit(signal, b_values=B_VALUES):
    """Fit log(S) = intercept + slope*b by independent float64 lstsq.

    Codes: 0 valid; 1 finite nonpositive input; 2 ADC <= 1e-12;
    3 nonfinite input or fitted ADC. Nonfinite takes precedence over code 1.
    """
    values = np.asarray(signal, dtype=np.float64)
    b = np.asarray(b_values, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] != 4:
        raise ValueError("signal must have leading dimension 4")
    if b.shape != (4,) or not np.array_equal(np.sort(b), B_VALUES):
        raise ValueError("exactly b=0,100,600,800 is required")
    flat = values.reshape(4, -1)
    adc = np.full(flat.shape[1], np.nan, dtype=np.float64)
    reason = np.full(flat.shape[1], 3, dtype=np.uint8)
    finite = np.isfinite(flat).all(axis=0)
    nonpositive = finite & (flat <= 0).any(axis=0)
    adc[nonpositive] = 0.0
    reason[nonpositive] = 1
    eligible = finite & ~nonpositive
    if eligible.any():
        design = np.column_stack((np.ones(4, dtype=np.float64), b))
        coefficients = np.linalg.lstsq(design, np.log(flat[:, eligible]), rcond=None)[0]
        fitted = -coefficients[1]
        valid_fit = np.isfinite(coefficients).all(axis=0) & np.isfinite(fitted)
        nonpositive_fit = valid_fit & (fitted <= 1e-12)
        positive_fit = valid_fit & ~nonpositive_fit
        locations = np.flatnonzero(eligible)
        adc[locations[nonpositive_fit]] = 0.0
        reason[locations[nonpositive_fit]] = 2
        adc[locations[positive_fit]] = fitted[positive_fit]
        reason[locations[positive_fit]] = 0
    shape = values.shape[1:]
    adc = adc.reshape(shape)
    return {"adc": adc, "reason": reason.reshape(shape), "high": adc > 0.01}


def from_datasets(dslist):
    """Independently decode and sort classic single-frame MR TRACE datasets.

    Only explicit public DiffusionBValue is accepted. Public rescale slope and
    intercept are applied once when present; both absent means unchanged stored
    signal, while a partially missing pair is rejected.
    positions are raw three-dimensional IPPs; projected_positions are signed
    millimetre projections of IPP on IOP's cross product, used for sorting.
    """
    datasets = list(dslist)
    if not datasets:
        raise ValueError("empty TRACE dataset collection")
    rows = []
    orientation = None
    spacing = None
    image_shape = None
    for ds in datasets:
        if str(getattr(ds, "SOPClassUID", "")) != "1.2.840.10008.5.1.4.1.1.4":
            raise ValueError("only classic MR Image Storage is supported")
        if int(getattr(ds, "NumberOfFrames", 1)) != 1:
            raise ValueError("multiframe images are not classic single-frame TRACE")
        b = float(ds.DiffusionBValue)
        if b not in B_VALUES:
            raise ValueError("unexpected or nonfinite explicit public b value")
        iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
        ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        pixel_spacing = np.asarray(ds.PixelSpacing, dtype=np.float64)
        if (iop.shape != (6,) or ipp.shape != (3,) or pixel_spacing.shape != (2,)
                or not np.isfinite(iop).all() or not np.isfinite(ipp).all()
                or not np.isfinite(pixel_spacing).all() or (pixel_spacing <= 0).any()):
            raise ValueError("invalid orientation, position or pixel spacing")
        if (abs(np.linalg.norm(iop[:3]) - 1) > 1e-5
                or abs(np.linalg.norm(iop[3:]) - 1) > 1e-5
                or abs(np.dot(iop[:3], iop[3:])) > 1e-5):
            raise ValueError("IOP must contain orthonormal direction cosines")
        if orientation is None:
            orientation, spacing = iop, pixel_spacing
        elif (not np.allclose(iop, orientation, rtol=0, atol=1e-5)
                or not np.allclose(pixel_spacing, spacing, rtol=0, atol=1e-5)):
            raise ValueError("inconsistent TRACE orientation or spacing")
        if any(el.tag in (0x00409096, 0x00283000) for el in ds.iterall()):
            raise ValueError("alternative TRACE transformation needs explicit review")
        have_slope, have_intercept = hasattr(ds, "RescaleSlope"), hasattr(ds, "RescaleIntercept")
        if not have_slope and not have_intercept:
            slope, intercept = 1.0, 0.0  # Independently retain untransformed stored signal.
        elif have_slope and have_intercept:
            slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)
        else:
            raise ValueError("partially missing public rescale metadata")
        if not np.isfinite([slope, intercept]).all() or slope <= 0:
            raise ValueError("invalid public rescale metadata")
        pixels = np.asarray(ds.pixel_array)
        if pixels.ndim != 2:
            raise ValueError("expected a two-dimensional classic MR pixel array")
        if image_shape is None:
            image_shape = pixels.shape
        elif pixels.shape != image_shape:
            raise ValueError("inconsistent TRACE image dimensions")
        with np.errstate(over="ignore", invalid="ignore"):
            decoded = pixels.astype(np.float64) * slope + intercept
        # Use one consistent independently established normal for every image.
        position = float(np.dot(ipp, np.cross(orientation[:3], orientation[3:])))
        rows.append((b, position, ipp, decoded))
    volumes = []
    base_positions = None
    base_ipps = None
    for b in B_VALUES:
        group = sorted((row for row in rows if row[0] == b), key=lambda row: row[1])
        if not group:
            raise ValueError("missing b group")
        positions = np.asarray([row[1] for row in group], dtype=np.float64)
        ipps = np.asarray([row[2] for row in group], dtype=np.float64)
        if np.any(np.diff(positions) <= 0.01):
            raise ValueError("duplicate or unresolved slice positions")
        if base_positions is None:
            base_positions, base_ipps = positions, ipps
        elif (positions.shape != base_positions.shape
                or not np.allclose(ipps, base_ipps, rtol=0, atol=0.01)):
            raise ValueError("missing or inconsistent slice positions across b values")
        volumes.append(np.stack([row[3] for row in group]))
    return {"signal": np.stack(volumes), "positions": base_ipps,
            "projected_positions": base_positions,
            "b_values": np.asarray(B_VALUES, dtype=np.float64)}
