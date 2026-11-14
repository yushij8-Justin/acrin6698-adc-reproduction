# Original ACRIN-6698 baseline analysis

Run this module through the repository entry point or directly with the pinned environment:

```sh
python original30/run.py --cache /path/to/verified-archives --output /path/to/new-output
```

The 30-pair imaging manifest gives the five TCIA Series Instance UIDs for each FIRST or REPEAT scan. It uses shuffled participant labels that differ from the study figures. The pipeline uses the acquisition role from this manifest and checks that DICOM patient, study, time point and series identities agree.

The frozen core applies published ADC scales, aligns MASK and SEG to the ADC grid, checks their common support, decodes classic MR TRACE at the four specified b values, and fits ADC. The original reference rule is applied to ADC, MASK, SEG and TRACE objects using the five planned series. The 18 reference holds contain unresolved MR references. Six of them also contain an unresolved Modality Performed Procedure Step (MPPS) reference. Twelve contain a Detached Study Management reference to an object outside the set, a class the original gate ignored. The independent solver verifies the fit and disposition of every ROI voxel in technically evaluable acquisitions. Averages include zeros. Both accepted scans are required for paired summaries. The fixed minimum 24 pairs for inference is above the 17 pairs available here. Site membership is unnecessary for these descriptive summaries. The archived number of evaluable sites is supplied as an aggregate.

`summary.json` reproduces Table 1, Table 2, and the additional numerical descriptions in Results. The runner verifies its values against `expected_summary.json` before writing outputs. `figure_2_points.local.csv` gives the numerical data for Figure 2 with this release’s participant labels. `technical_readouts.local.json` contains the local acquisition results. The original Table 3 and the visual-review branch of Figure 1 describe collaborator-reported physician confirmation of archived categories. Their aggregate counts appear in `expected_summary.json`. Computational replay cannot regenerate human visual ratings.

The file-level integrity check accepts archived or newly retrieved ZIP packaging when the DICOM payloads match the frozen series checksums. The runner uses the provided imaging cache and reads no clinical or outcome files. The historical provenance and reviewer-confirmed classifications are distinct from this software replay.
