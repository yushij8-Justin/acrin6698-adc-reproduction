# Input inventory and retrieval

The inventories contain the public imaging SeriesInstanceUID, instance count, state and the expected content fingerprints. Pairing maps under `original30` and `holdout18` associate each series with the anonymous release pair, scan role and series kind. The five series roles per scan are ADC, MASK, SEG, TRACE and SOURCE. The pair labels serve this package only.

| Input | Series specified | Series verified and used as inputs | DICOM objects in verified series |
|---|---:|---:|---:|
| Original 30 pairs | 300 | 300 | 24,586 |
| Hold-out 18 pairs | 180 | 170 | 14,245 |

The transport inventories and generation-pinned IDC locator are bound by `manifests/integrity.json`. Verify the input pairing and software snapshot against the release checksums before running. Each series fingerprint is order independent. It comprises the NBIA-style MD5 of concatenated sorted lowercase per-instance MD5 hex strings and, for each available series, a SHA-256 of concatenated sorted lowercase per-instance SHA-256 hex strings. This verifies the DICOM bytes independently of ZIP packaging. Duplicates contribute their multiplicity. DICOM identifiers and metadata are verified locally by the cohort runners.

### Where the files come from

- **Original.** The workflow requests the listed series from the public TCIA NBIA endpoint. The manifest was bound to checksummed historical TCIA ZIPs from the original acquisition and replay.
- **Hold-out.** The IDC locator lists only the 170 historically available series, with the public object path, exact generation, size and MD5. All frozen series inventories were checked against their NBIA MD5 hierarchy at execution. The available series SHA-256 fingerprints were computed from the verified historical DICOM. A TCIA download is also accepted if it has exactly the same DICOM payload.
- **Historical exhausted status.** The 10 remaining hold-out series have a historical exclusion status. This describes the original transfer attempts rather than current public availability. These series are never requested by this replay and never contribute pixels, even if supplied in the cache.

The fetcher uses public imaging endpoints with HTTPS, needs no credentials, and checks file sizes, ZIP member names, instance counts and contents before accepting a series. For slow networks, reduce concurrency with `--workers 2`. Each historically available series missing from the local cache has up to three transport attempts per invocation. These attempts reconstruct the fixed input cache and are separate from the historical study's transfer records. The software never substitutes another acquisition.

Sources: [TCIA ACRIN-6698 collection and attribution](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=50135447), [TCIA API guide](https://wiki.cancerimagingarchive.net/pages/viewpage.action?navigatingVersions=true&pageId=477659147), [NCI IDC versioning](https://learn.canceridc.dev/data/data-versioning) and [NCI IDC object identifiers](https://learn.canceridc.dev/data/organization-of-data/guids-and-uuids).
