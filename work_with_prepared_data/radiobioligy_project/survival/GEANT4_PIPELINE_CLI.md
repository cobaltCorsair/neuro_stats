# GEANT4 Pipeline CLI

This guide covers the command-line wrapper around
[pipeline_geant4_to_prediction.py](/C:/dev/neuro_stats/work_with_prepared_data/radiobioligy_project/survival/pipeline_geant4_to_prediction.py).

On Windows, the recommended entry point is:

```bat
run_geant4_pipeline.bat
```

The launcher lives next to the pipeline script and forwards all CLI arguments.
It first tries `PYTHON_EXE`, then `.venv`, then Poetry virtualenvs, then `py -3.10`, then `python`.

If several Python environments are installed, you can pin the interpreter explicitly:

```bat
set PYTHON_EXE=C:\path\to\python.exe
run_geant4_pipeline.bat --help
```

## Required Inputs

Minimal inputs:

- `dose.pb`
  - `totDoseVoxelMap` for single-field mode
  - `fullVoxelMap` for mixed-field mode
- `geometry.ivz`
  - serialized `InputVoxelMap`

Optional inputs:

- `contour.pb`
  - serialized `ContourMeta`
- `tumor_mask.nii` or `tumor_mask.nii.gz`
  - binary 3D Slicer NIfTI mask already resampled to the GEANT4 voxel grid
- `fit_results.csv`
  - summary CSV exported from `fit_alpha_beta_using_processor.py`

## Single-Field Example

Use fixed radiobiological parameters:

```bat
run_geant4_pipeline.bat ^
  C:\path\to\dose.pb ^
  C:\path\to\geometry.ivz ^
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
  C:\path\to\dose.pb ^
  C:\path\to\geometry.ivz ^
  --let-alpha0 0.08 ^
  --let-lambda-alpha 0.003 ^
  --let-beta0 0.02 ^
  --let-lambda-beta 0.0 ^
  --n-fractions 5 ^
  --output-dir C:\path\to\prediction_output
```

## Mixed-Field Example

Use `fullVoxelMap` and map each GEANT4 component to a radiobiological family:

```bat
run_geant4_pipeline.bat ^
  C:\path\to\fullVoxelMap.pb ^
  C:\path\to\geometry.ivz ^
  --contour-path C:\path\to\contour.pb ^
  --fit-results-csv C:\path\to\fit_results.csv ^
  --mixed-field ^
  --component-family-map protonDose=p,mainDose=y,midDose=n,stuffDose=e ^
  --schedule-days 0,1,2,3,4 ^
  --output-dir C:\path\to\prediction_output
```

Current limitation:

- NIfTI contour import expects one binary mask per target structure.
- The mask must already match the GEANT4 grid shape `(xLen, yLen, zLen)`.
- Multi-label NIfTI segmentations are not supported yet.

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

- `--schedule-days 0,1,2,3,4`
- `--n-fractions 5`
- `--growth-duration-days 60`
- `--growth-time-step-days 0.25`
- `--model-kind classic_lq`
- `--model-kind repair_lq`
- `--repair-half-time-hours 1.0`
- `--bed-eqd2-fractions 1,3,5,10,20,30`

## Help

```bat
run_geant4_pipeline.bat --help
```
