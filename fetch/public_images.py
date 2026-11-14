"""Download the exact public DICOM input series to a local cache."""
from __future__ import annotations

import csv
import gzip
import hashlib
import http.client
import io
import json
import os
import re
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit

from fetch.image_io import (
    AVAILABLE, ROOT, InputError, archive_path, check_archive, inventory, require,
)

NBIA_HOST = "public.cancerimagingarchive.net"
IDC_HOST = "storage.googleapis.com"
NBIA = f"https://{NBIA_HOST}/nbia-api/services/v4/getImageWithMD5Hash"
NBIA_ONLY_LIMIT = 4 * 1024**3
OBJECT_KEY = re.compile(r"[0-9a-f-]{36}/[0-9a-f-]{36}\.dcm\Z")


class BoundedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        require(target.scheme == "https" and target.hostname == NBIA_HOST and
                target.port in (None, 443) and not target.username and not target.password and
                target.path == "/nbia-api/services/v4/getImageWithMD5Hash" and
                target.query == urlsplit(req.full_url).query,
                "UNEXPECTED_REDIRECT")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_nbia(entry, target, timeout):
    url = NBIA + "?" + urlencode({"SeriesInstanceUID": entry["uid"]})
    opener = urllib.request.build_opener(BoundedRedirect())
    request = urllib.request.Request(url, headers={"User-Agent": "acrin6698-reproduction/1"})
    with opener.open(request, timeout=timeout) as response, target.open("wb") as output:
        count = 0
        while data := response.read(1024 * 1024):
            count += len(data)
            require(count <= NBIA_ONLY_LIMIT, "NBIA_ARCHIVE_TOO_LARGE")
            output.write(data)


def locator_for(entries):
    path = ROOT / "manifests" / "holdout18_idc_v25.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        locator = json.load(handle)
    require(locator["idc_version"] == 25 and locator["bucket"] == "idc-open-data",
            "UNEXPECTED_IDC_SOURCE")
    require(locator["object_columns"] == ["name", "generation", "size", "md5"], "INVALID_IDC_COLUMNS")
    objects = locator["objects"]
    require(set(objects) == {uid for uid, row in entries.items() if row["state"] == AVAILABLE},
            "IDC_LIST_DIFFERS_FROM_HISTORICAL_AVAILABLE_SET")
    for uid, files in objects.items():
        require(len(files) == entries[uid]["instances"] and len({row[0] for row in files}) == len(files),
                "BAD_IDC_INSTANCE_COUNT")
        for key, generation, size, md5 in files:
            require(OBJECT_KEY.fullmatch(key) is not None and generation.isascii() and generation.isdecimal() and
                    type(size) is int and 0 < size <= 256 * 1024**2 and re.fullmatch(r"[a-f0-9]{32}", md5),
                    "INVALID_IDC_OBJECT")
        series_md5 = hashlib.md5("".join(sorted(row[3] for row in files)).encode("ascii"),
                                 usedforsecurity=False).hexdigest()
        require(series_md5 == entries[uid]["series_md5"], "IDC_INVENTORY_NOT_EQUIVALENT_TO_TCIA")
    return objects


def fetch_idc(objects, target, timeout):
    connection = http.client.HTTPSConnection(IDC_HOST, timeout=timeout)
    checksums = io.StringIO()
    writer = csv.writer(checksums, lineterminator="\n")
    try:
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
            for idx, (name, generation, size, md5) in enumerate(objects, 1):
                path = f"/idc-open-data/{quote(name, safe='/')}?generation={generation}"
                # A dropped reusable connection is retried once at file level.
                for network_attempt in range(2):
                    try:
                        connection.request("GET", path, headers={"User-Agent": "acrin6698-reproduction/1"})
                        with connection.getresponse() as response:
                            require(response.status == 200, f"IDC_HTTP_{response.status}")
                            observed_generation = response.getheader("x-goog-generation")
                            require(observed_generation == generation, "IDC_GENERATION_CHANGED")
                            content = response.read(size + 1)
                        break
                    except (OSError, http.client.HTTPException):
                        connection.close()
                        if network_attempt:
                            raise
                require(len(content) == size and
                        hashlib.md5(content, usedforsecurity=False).hexdigest() == md5,
                        "IDC_OBJECT_CONTENT_CHANGED")
                filename = f"image_{idx:05d}.dcm"
                archive.writestr(filename, content)
                writer.writerow((filename, md5))
            archive.writestr("md5hashes.csv", checksums.getvalue())
    finally:
        connection.close()


def ensure_one(cache, entry, provider, objects, timeout):
    destination = archive_path(cache, entry["uid"])
    require(entry["state"] == AVAILABLE, "ATTEMPT_TO_FETCH_HISTORICALLY_EXCLUDED_SERIES")
    if destination.exists():
        check_archive(destination, entry)
        return "cache"
    for attempt in range(3):
        handle, temp_name = tempfile.mkstemp(dir=cache, prefix=".transfer-", suffix=".part")
        os.close(handle)
        temp = Path(temp_name)
        try:
            if provider == "idc":
                fetch_idc(objects[entry["uid"]], temp, timeout)
            else:
                fetch_nbia(entry, temp, timeout)
            check_archive(temp, entry)
            # Publish a fully verified archive atomically without replacing existing data.
            try:
                os.link(temp, destination)
            except FileExistsError:
                check_archive(destination, entry)
            return "fetched"
        except (OSError, http.client.HTTPException):
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))
        finally:
            temp.unlink(missing_ok=True)
    raise InputError("IMAGE_TRANSFER_FAILED")


def fetch(cohort, cache, provider, workers, timeout):
    entries = inventory(cohort)
    selected = [row for row in entries.values() if row["state"] == AVAILABLE]
    if provider == "auto":
        provider = "idc" if cohort == "holdout18" else "nbia"
    require(provider in {"idc", "nbia"} and (provider != "idc" or cohort == "holdout18"),
            "NO_PINNED_IDC_INVENTORY_FOR_SELECTED_COHORT")
    objects = locator_for(entries) if provider == "idc" else None
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)

    def process(entry):
        try:
            return ensure_one(cache, entry, provider, objects, timeout), None
        except InputError as exc:
            return None, str(exc)
        except (OSError, http.client.HTTPException) as exc:
            return None, type(exc).__name__

    totals = {"fetched": 0, "cache": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i, (status, failure) in enumerate(executor.map(process, selected), 1):
            if failure:
                totals["failed"] += 1
                print(f"Series position {i}: {failure}", flush=True)
            else:
                totals[status] += 1
            if i % 20 == 0 or i == len(selected):
                print(f"{cohort} {i}/{len(selected)} archives checked or transferred", flush=True)
    if totals["failed"]:
        raise InputError("One or more expected archives could not be obtained and verified. The historical sample is unchanged.")
    return {"cohort": cohort, "provider": provider, "historically_excluded": len(entries) - len(selected), **totals}
