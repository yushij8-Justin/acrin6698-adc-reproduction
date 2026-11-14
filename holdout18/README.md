# Reproduce the ACRIN-6698 hold-out analysis

The 18 same-archive baseline scan–rescan pairs are analyzed separately. The recorded input set contains five series per scan, keeps all 180 series in the attempted denominator, and has 170 historically verified series. The 10 series whose original transfer attempts were exhausted remain excluded from the replay, including when there is a newer copy of a series. A missing or changed copy of any of the 170 verified series stops the replay. This preserves the distinction between the historical disposition and a local retrieval failure.

The package root README describes retrieval and the pinned Python environment. With the verified archives in a local cache directory, run this command from the package root:

```sh
python holdout18/reproduce.py --cache /path/to/holdout-cache --output /path/to/new-holdout-output
```

The cache uses one ZIP per series with the public series UID as the filename. The shared reader verifies instance count, NBIA MD5 and the frozen payload SHA-256. The files are read locally. The role map links only the public imaging series UIDs to fixed, opaque pair labels, acquisition roles, manufacturer, and combined-source flags. The deidentified subject and study DICOM fields are checked in memory for series and pair consistency. The aggregate output contains neither values for individual subjects nor identity fields. No outcome tables or other clinical files are inputs.

## Calculations and outputs

The strict path checks the fixed source-reference graph, TRACE mapping, ADC geometry and units, and agreement between the released mask and SEG. For an accepted scan, it fits the four b values to derive TRACE ADC and compares the resulting ROI mean against the released ADC ROI mean. It also checks agreement with an independent least-squares fit in the ROI. The fixed source-form flags provide context and do not override the provenance rule. A comparison may use a verifiable linkage between the released derived objects without a direct link to an original scanner series.

For pairs with both scans accepted, the difference (released ADC minus recalculated ADC) is averaged within each pair and then across equally weighted patients. Within-subject CV uses the log repeat/first ratio for each method. Output fields are labeled in the same units as the public result table and checked against the displayed published precision before writing `summary.json`. The historical recovery and mapped-versus-stored contrast are represented as unavailable estimates. The 0.05 × 10⁻³ mm²/s engineering margin applies to the mapped-versus-stored sensitivity comparison.

The replay recalculates the strict-path cohort aggregates from frozen DICOM content. Its evidence-source scope is the fixed objects used for the strict analysis. The extended archival metadata lookup, independent human adjudication, and recovery pathway are outside this replay. `published_strict_results.tsv` is a reference for the final checks, and `workflow_provenance.json` records the original verified computation modules, role map, and historical provenance digests.
