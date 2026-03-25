# GEANT4 Pipeline CLI

This guide covers the command-line wrapper around
`pipeline_geant4_to_prediction.py`.

On Windows, the recommended entry point is:

```bat
run_geant4_pipeline.bat
```

The launcher forwards all arguments to the pipeline CLI.
It first tries `PYTHON_EXE`, then `.venv`, then Poetry virtualenvs, then
`py -3.10`, then `python`.

If several Python environments are installed, you can pin the interpreter
explicitly:

```bat
set PYTHON_EXE=C:\path\to\python.exe
run_geant4_pipeline.bat --help
```

For backend/API details, see [TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md).

## Supported Input Modes

The first positional argument may be one of:

- protobuf single-field dose map
  - `totDoseVoxelMap.pb`
- protobuf mixed-field dose map
  - `fullVoxelMap.pb`
- DICOM RT Dose file
  - `RT Dose .dcm`
- one-animal input directory
  - a folder that contains one case and can be autodiscovered

The second positional argument is optional:

- `geometry.ivz` or `InputVoxelMap.pb`
- required for explicit protobuf input when geometry cannot be autodiscovered
- omitted for `RT Dose`

Optional contour inputs:

- `ContourMeta.pb`
- aligned binary NIfTI mask: `.nii` or `.nii.gz`
- `RTSTRUCT .dcm`

Optional radiobiology input sources:

- `--fit-results-csv`
- `--alpha` with `--beta`
- `--let-alpha0` with `--let-beta0`

## What Autodiscovery Does

If the first argument is a directory, the pipeline tries to find inputs inside it.

Typical folder-mode use case:

- one folder per animal
- one `RT Dose`
- one `RTSTRUCT`

Autodiscovery behavior:

- for DICOM mode, it looks for one `RT Dose` and one `RTSTRUCT`
- for protobuf mode, it looks for one dose input and one geometry input
- if several candidates are found, the pipeline stops and asks you to pass explicit paths instead

## Protobuf Single-Field Example

Use fixed radiobiological parameters:

```bat
run_geant4_pipeline.bat ^
  C:\path\to\totDoseVoxelMap.pb ^
  C:\path\to\InputVoxelMap.ivz ^
  --contour-path C:\path\to\tumor_mask.nii.gz ^
  --alpha 0.10 ^
  --beta 0.02 ^
  --schedule-days 0,1,2,3,4 ^
  --structure-name tumor ^
  --output-dir C:\path\to\prediction_output
```

Use an explicit LET profile instead of constant `alpha/beta`:

```bat
run_geant4_pipeline.bat ^
  C:\path\to\totDoseVoxelMap.pb ^
  C:\path\to\InputVoxelMap.ivz ^
  --let-alpha0 0.08 ^
  --let-lambda-alpha 0.003 ^
  --let-beta0 0.02 ^
  --let-lambda-beta 0.0 ^
  --n-fractions 5 ^
  --output-dir C:\path\to\prediction_output
```

## RT Dose + RTSTRUCT Example

Use DICOM dose and structure directly:

```bat
run_geant4_pipeline.bat ^
  C:\path\to\dose.dcm ^
  --contour-path C:\path\to\struct.dcm ^
  --alpha 0.10 ^
  --beta 0.02 ^
  --structure-name PTV_High ^
  --schedule-days 0 ^
  --output-dir C:\path\to\prediction_output
```

Notes:

- geometry is omitted for DICOM mode
- for `RTSTRUCT`, use the exact ROI name in `--structure-name`
- if no contour is passed for `RT Dose`, the pipeline uses all nonzero dose voxels as a fallback ROI

## Folder-Mode Example

Use one animal folder instead of explicit DICOM files:

```bat
run_geant4_pipeline.bat ^
  C:\path\to\animal_folder ^
  --alpha 0.10 ^
  --beta 0.02 ^
  --structure-name PTV_High ^
  --schedule-days 0 ^
  --output-dir C:\path\to\animal_folder\prediction_output
```

This is the preferred DICOM workflow when:

- each animal has its own calculation directory
- the directory contains exactly one `RT Dose`
- the directory contains exactly one matching `RTSTRUCT`

## Mixed-Field Protobuf Example

Use `fullVoxelMap` and map each GEANT4 component to a radiobiological family:

```bat
run_geant4_pipeline.bat ^
  C:\path\to\fullVoxelMap.pb ^
  C:\path\to\InputVoxelMap.ivz ^
  --contour-path C:\path\to\contour.pb ^
  --fit-results-csv C:\path\to\fit_results.csv ^
  --mixed-field ^
  --component-family-map protonDose=p,mainDose=y,midDose=n,stuffDose=e ^
  --schedule-days 0,1,2,3,4 ^
  --output-dir C:\path\to\prediction_output
```

If `--component-family-map` is omitted, the pipeline uses the default mapping:

- `protonDose -> p`
- `midDose -> n`
- `mainDose -> y`
- `stuffDose -> e`

## Outputs

The pipeline writes:

- `dvh_<structure>.csv`
- `sf_per_voxel.csv`
- `aggregated_params.json`
- `growth_curve.csv`
- `bed_eqd2_table.csv`
- `summary.json`

The CLI also prints:

- `mean_dose_gy`
- `mean_sf`
- `effective_alpha`
- `effective_beta`
- `summary_json`

## Radiobiology Source Options

Provide exactly one of these:

- `--fit-results-csv`
- `--alpha` and `--beta`
- `--let-alpha0` and `--let-beta0`

## Useful Options

- `--contour-path`
- `--structure-name`
- `--schedule-days 0,1,2,3,4`
- `--n-fractions 5`
- `--growth-duration-days 60`
- `--growth-time-step-days 0.25`
- `--model-kind classic_lq`
- `--model-kind repair_lq`
- `--repair-half-time-hours 1.0`
- `--bed-eqd2-fractions 1,3,5,10,20,30`

## Current Limitations

- `RT Dose` support is currently single-field only.
- mixed-field remains protobuf-only.
- NIfTI import expects one binary mask per target structure.
- the mask must already match the target voxel grid shape.
- multi-label NIfTI segmentations are not supported.
- folder mode assumes one case per directory.

## Help

```bat
run_geant4_pipeline.bat --help
```
