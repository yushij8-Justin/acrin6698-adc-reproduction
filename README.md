# Reproduce the ACRIN-6698 breast diffusion MRI analysis

This package reconstructs the quantitative results of *Traceable Reuse of Released Breast Diffusion MRI for ADC Measurement in ACRIN-6698* from fixed public imaging series. The original 30 baseline scan–rescan pairs and the 18 same-archive hold-out pairs have separate input inventories and separate analyses. The workflow accesses public DICOM only and does not read clinical or treatment outcome tables.

## Installation

Use Python 3.12.14. From this directory, create an isolated environment and install the pinned versions of NumPy and pydicom.

```sh
python3.12 --version
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Before running, verify `SHA256SUMS` from the package root with `shasum -a 256 -c SHA256SUMS` (macOS) or `sha256sum -c SHA256SUMS` (Linux).

The instructions use a POSIX shell. On Windows, substitute `.venv\Scripts\python.exe`. The download service and image data stay outside the repository. Plan for at least 6 GB of free space if downloading both cohorts and keeping both result directories.

## Reproduce from DICOM

Run each cohort independently. Use distinct cache and result directories.

```sh
.venv/bin/python -B replicate.py all original30 --cache cache/original30 --output outputs/original30
.venv/bin/python -B replicate.py all holdout18 --cache cache/holdout18 --output outputs/holdout18
```

`all` first retrieves or checks the available DICOM archives and then runs the analysis. Use `fetch` to obtain and check archives without fitting ADC. Use `analyze` with the same cohort, `--cache` and a new `--output` to replay from a verified cache without network access. Each cached ZIP is named by its exact public SeriesInstanceUID, followed by `.zip`. Provide ZIPs containing the original DICOM bytes. ZIP member names and compression can differ. Avoid converting to NIfTI, resaving DICOM, editing tags or regenerating ADC maps before running the analysis.

For the original cohort, the default retrieval provider is the public TCIA NBIA v4 API. For the hold-out cohort, the default provider is NCI Imaging Data Commons (IDC) v25 using fixed cloud-object generations. Each IDC series has the same DICOM content as the corresponding TCIA release used for verification. To use TCIA for the hold-out instead, add `--provider nbia`. Both routes check the frozen payload fingerprints. Data already fetched with TCIA's supported download tools may be used after arranging the unmodified series into the required ZIPs. See [input and retrieval specification](docs/inputs.md).

An expected archive that is missing or has different content stops the replay. This ensures that a new retrieval failure cannot change the study denominator or produce an apparently successful replay with a smaller sample. An interrupted transfer can be resumed by repeating the command with the same cache. If an existing ZIP fails verification, move it aside and obtain the original DICOM bytes before retrying.

## Results

| Analysis | Fixed attempted pairs | Pairs with both scans eligible | Mean released minus recalculated ADC (10⁻³ mm²/s) | Released and recalculated ADC wCV |
|---|---:|---:|---:|---|
| Original cohort | 30 | 17 | +0.028925 | 4.304%, 4.104% |
| Same-archive hold-out | 18 | 6 | +0.023468 | 5.140%, 4.899% |

Each result directory contains a computed summary checked against the relevant archived numerical targets. In the hold-out, 170 of the 180 specified series were available historically. The 10 series that reached the original transfer limit are marked in the inventory and excluded by the replay. Both the recovery rate and the difference between ADC from mapped and stored TRACE signals remain unavailable estimates. The engineering margin of 0.05 × 10⁻³ mm²/s refers to that unavailable mapping comparison. The evaluable pairs in both cohorts were scanned on GE systems.

See [original cohort outputs](original30/README.md) and [hold-out cohort outputs](holdout18/README.md). The original physician review categories and their aggregate cross-tabulation are historical human assessments. The replay reproduces the computational measurements and uses the historical aggregate for the visual review branch. Read the [reproduction scope](docs/scope.md) for what is computed from DICOM and what is reported from a historical record.

## Digital reference tests

The `digital_reference` folder contains the original synthetic test archive. Extract it to a fresh location, then enter the extracted `adc_process_audit` directory and run the following with the pinned interpreter:

```sh
python run_dro.py --output out_dro
python examples/synthetic_pipeline.py --output out_example
```

The synthetic suite checks 21 cases and 299 assertions. The archive is preserved byte for byte as a historical witness. Its internal publication and licensing notes describe its original local status. The top-level release information applies to the current package.

## Software license

Version 1.0.0. The project software is distributed under the [MIT License](LICENSE), copyright Yushi Jia (贾玉石) and Bo Zhang (张博).

## Data attribution

The ACRIN-6698 images and segmentations are distributed under CC BY 4.0. They are retrieved from the data archive and are not redistributed here. The public imaging series UIDs in this package can be resolved to their public DICOM headers. Please observe the archive's data use policy. Cite the collection, its source publications and this package version. See [dataset and citation information](CITATION.md).
