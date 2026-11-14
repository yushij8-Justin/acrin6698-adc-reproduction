"""Load only a frozen imaging inventory and validate exact archived payloads."""
from __future__ import annotations

import hashlib
import io
import json
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"original30": (300, 300), "holdout18": (180, 170)}
AVAILABLE = "AVAILABLE"
EXHAUSTED = "HISTORICAL_EXHAUSTED"
UID = re.compile(r"[0-9]+(?:\.[0-9]+)+\Z")
MD5 = re.compile(r"[a-f0-9]{32}\Z")
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
NOTICE = {"md5hashes.csv", "license", "license.txt", "tcia_license.txt", "license.pdf", "readme.txt"}


class InputError(RuntimeError):
    """An input was absent, changed, or did not match the imaging inventory."""


def require(condition, code):
    if not condition:
        raise InputError(code)


def is_uid(value):
    return isinstance(value, str) and len(value) <= 64 and UID.fullmatch(value) is not None


def inventory(cohort):
    require(cohort in COUNTS, "UNKNOWN_COHORT")
    recipe = ROOT / "manifests" / f"{cohort}.json"
    data = json.loads(recipe.read_text(encoding="utf-8"))
    require(data["cohort"] == cohort and data["collection_doi"].lower() == "10.7937/tcia.kk02-6d95",
            "INPUT_MANIFEST_COLLECTION")
    rows = data["series"]
    index = {}
    for row in rows:
        require(set(row) == {"uid", "instances", "series_md5", "payload_sha256", "state"},
                "UNEXPECTED_MANIFEST_FIELD")
        uid = row["uid"]
        require(is_uid(uid) and uid not in index and type(row["instances"]) is int and row["instances"] > 0,
                "INVALID_SERIES_ENTRY")
        require(MD5.fullmatch(row["series_md5"]) is not None, "INVALID_EXPECTED_MD5")
        if row["state"] == AVAILABLE:
            require(isinstance(row["payload_sha256"], str) and SHA256.fullmatch(row["payload_sha256"]) is not None,
                    "MISSING_FROZEN_SHA256")
        else:
            require(row["state"] == EXHAUSTED and row["payload_sha256"] is None, "INVALID_SERIES_STATE")
        index[uid] = row
    require((len(index), sum(x["state"] == AVAILABLE for x in index.values())) == COUNTS[cohort],
            "WRONG_FROZEN_SERIES_COUNTS")
    seal = json.loads((ROOT / "manifests" / "integrity.json").read_text(encoding="utf-8"))
    require(hashlib.sha256(recipe.read_bytes()).hexdigest() == seal[recipe.name], "MANIFEST_HASH_CHANGED")
    return index


def archive_path(cache, uid):
    require(is_uid(uid), "INVALID_SERIES_UID")
    return Path(cache) / (uid + ".zip")


def hash_multisets(md5s, sha256s):
    """NBIA's series MD5 and a stronger order-independent DICOM payload hash."""
    return (
        hashlib.md5("".join(sorted(md5s)).encode("ascii"), usedforsecurity=False).hexdigest(),
        hashlib.sha256("".join(sorted(sha256s)).encode("ascii")).hexdigest(),
    )


def archive_payload(path, expected_count, keep=False):
    md5s, sha256s, payload = [], [], []
    try:
        with zipfile.ZipFile(path) as bundle:
            names = set()
            dicoms = []
            for item in bundle.infolist():
                name = PurePosixPath(item.filename)
                require(not name.is_absolute() and ".." not in name.parts and "\\" not in item.filename and
                        item.filename not in names and not stat.S_ISLNK(item.external_attr >> 16),
                        "UNSAFE_ZIP_ENTRY")
                names.add(item.filename)
                if item.is_dir():
                    continue
                if name.suffix.lower() == ".dcm":
                    require(0 < item.file_size <= 256 * 1024**2, "OBJECT_SIZE_OUTSIDE_LIMIT")
                    dicoms.append(item)
                else:
                    require(name.name.lower() in NOTICE, "UNEXPECTED_NONIMAGE_ZIP_MEMBER")
            require(len(dicoms) == expected_count, "DICOM_INSTANCE_COUNT_MISMATCH")
            for entry in dicoms:
                # This reads the entire member and checks its CRC before hashing the bytes.
                content = bundle.read(entry)
                md5s.append(hashlib.md5(content, usedforsecurity=False).hexdigest())
                sha256s.append(hashlib.sha256(content).hexdigest())
                if keep:
                    payload.append(content)
    except (FileNotFoundError, OSError, zipfile.BadZipFile, NotImplementedError) as exc:
        raise InputError("MISSING_OR_UNREADABLE_DICOM_ARCHIVE") from exc
    return (*hash_multisets(md5s, sha256s), payload)


def check_archive(path, entry, keep=False):
    require(entry["state"] == AVAILABLE, "HISTORICALLY_UNAVAILABLE_SERIES")
    md5, sha256, payload = archive_payload(path, entry["instances"], keep)
    require((md5, sha256) == (entry["series_md5"], entry["payload_sha256"]), "DICOM_PAYLOAD_FINGERPRINT_MISMATCH")
    return payload


def read_verified_series(cache, entry):
    return check_archive(archive_path(cache, entry["uid"]), entry, keep=True)


def read_verified_datasets(cache, entry):
    import pydicom

    return [pydicom.dcmread(io.BytesIO(data)) for data in read_verified_series(cache, entry)]


def bindings_ok():
    """Verify packaged tables before running anything that accesses the network."""
    folder = ROOT / "manifests"
    seal = json.loads((folder / "integrity.json").read_text(encoding="utf-8"))
    fixed = {"original30.json", "holdout18.json", "holdout18_idc_v25.json.gz"}
    require(set(seal) == fixed, "INCOMPLETE_MANIFEST_SEAL")
    for name in sorted(fixed):
        path = folder / name
        require(hashlib.sha256(path.read_bytes()).hexdigest() == seal[name], "MANIFEST_HASH_CHANGED")
