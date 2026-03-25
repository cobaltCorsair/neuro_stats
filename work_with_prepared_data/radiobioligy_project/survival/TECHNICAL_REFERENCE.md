# Survival Technical Reference

This document is the canonical technical reference for the `survival` module.
It describes the current backend structure, public entry points, supported input
contracts, and output artifacts.

Use the other docs for different purposes:

- [USER_GUIDE_RU.md](USER_GUIDE_RU.md): operator guide and GUI workflows.
- [GEANT4_PIPELINE_CLI.md](GEANT4_PIPELINE_CLI.md): command-line usage for the pipeline.
- [TUMOR_GROWTH_PREDICTOR.md](TUMOR_GROWTH_PREDICTOR.md): predictor behavior and model details.
- [PLAN_ALPHA_BETA_EXTENSIONS.md](PLAN_ALPHA_BETA_EXTENSIONS.md): roadmap/history for fitter extensions.
- [PLAN_PREDICTIVE_MODEL.md](PLAN_PREDICTIVE_MODEL.md): roadmap/history for the predictive bridge.

The plan files remain useful for rationale and implementation history, but this
file should be treated as the current-state reference.

## Scope

The module contains three major product layers:

1. Radiobiology fitting from Excel tumor-volume series.
2. Tumor growth prediction from fitted or manual radiobiological parameters.
3. GEANT4 / RT Dose to voxel radiobiology to growth prediction pipeline.

## Module Map

### Fitter layer

- `fit_alpha_beta_using_processor.py`
  - core backend for fitting effective `alpha/beta` from in vivo volume-response data
  - supports multiple model families and scalar or full-curve response modes
  - exports summary CSV and structured results for GUI/tests
- `fit_alpha_beta_gui.py`
  - main fitter GUI window
  - launches the predictor window and the GEANT4 pipeline window
- `gui_csv_export.py`
  - helper exports used by the GUI layer

### Predictor layer

- `tumor_growth_predictor.py`
  - mathematical model for post-irradiation growth, repair-aware scheduling, and geometry scaling
- `tumor_growth_predictor_gui.py`
  - standalone predictor GUI
- `TUMOR_GROWTH_PREDICTOR.md`
  - user-facing description of the model and predictor window

### Pipeline layer

- `pipeline_geant4_to_prediction.py`
  - end-to-end orchestration from dose input to `CSV/JSON`
  - supports explicit file paths and input-directory autodiscovery
- `geant4_pipeline_gui.py`
  - GUI wrapper around the pipeline
- `run_geant4_pipeline.bat`
  - Windows launcher for the CLI
- `GEANT4_PIPELINE_CLI.md`
  - operator-oriented CLI guide

### Dose and contour ingestion layer

- `dose_reader.py`
  - unified read path for protobuf GEANT4 maps and DICOM RT Dose
- `nifti_contour_bridge.py`
  - aligned binary NIfTI mask to voxel-structure assignments
- `rtstruct_bridge.py`
  - RTSTRUCT rasterization onto the RT Dose grid
- `proto/`
  - protobuf schemas and generated Python bindings for GEANT4/NPLibrary data exchange

### Radiobiology aggregation and analysis layer

- `let_parametrization.py`
  - LET-dependent parameter container and fitting helpers
- `voxel_sf_calculator.py`
  - per-voxel `SF/BED` and structure-level aggregation
- `mixed_field_model.py`
  - mixed-field composition helpers
- `radiobiology_analysis.py`
  - `RBE`, `TCP`, `NTCP`, sensitivity analysis, and BED/EQD2 export

## Canonical Workflows

### 1. Fitter workflow

Input:

- `.xlsx` tumor-volume files
- control assignment, either pooled or explicit

Flow:

- parse Excel series and experiment metadata
- normalize treated curves against control
- derive scalar `SF` or full-curve response
- fit one or more model families
- export summaries and structured GUI results

Primary outputs:

- `LQFitResult`
- analysis summaries
- optional summary CSV

### 2. Predictor workflow

Input:

- treatment schedule as `TreatmentFraction` items
- `alpha/beta` from manual input or fitted runs
- optional geometry reference and control-derived growth parameters

Flow:

- compute per-fraction survival
- optionally accumulate incomplete repair using `Repair T1/2`
- evolve live and dead volume compartments
- reconstruct axis scaling for plots and 3D display

Primary outputs:

- `GrowthSimulationResult`
- time series for volume and geometry

### 3. GEANT4 / RT Dose pipeline workflow

Input modes:

- protobuf single-field: `totDoseVoxelMap` + `InputVoxelMap`
- protobuf mixed-field: `fullVoxelMap` + `InputVoxelMap`
- DICOM single-field: `RT Dose` + optional `RTSTRUCT`
- folder mode: one animal per folder, autodiscovered inputs

Flow:

- read dose grid into `DoseMap`
- resolve structure membership from `ContourMeta`, NIfTI, RTSTRUCT, or RT Dose fallback
- compute voxel-level `SF/BED`
- aggregate structure-level `alpha/beta`, `D90`, `D50`, `V20`, `EUD`
- pass aggregated radiobiology into the growth predictor
- export `CSV/JSON`

Primary outputs:

- `VolumetricSFResult`
- growth prediction
- pipeline output files written to disk

## Input Contracts

### Excel inputs for fitting and prediction

The fitter and predictor use the project-standard tumor Excel layout:

- row 1: experiment parameters
- row 2: day labels
- column 1: rat labels
- remaining cells: scalar volumes or `a-b-c` values, depending on workflow

This contract is handled by the processor layer outside this subtree and consumed
by the `survival` GUIs/backends.

### GEANT4 protobuf inputs

Single-field:

- dose map: `totDoseVoxelMap.pb`
- geometry: `InputVoxelMap.ivz` or `InputVoxelMap.pb`
- optional contour: `ContourMeta.pb`

Mixed-field:

- dose map: `fullVoxelMap.pb`
- geometry: `InputVoxelMap.ivz` or `InputVoxelMap.pb`
- optional contour: `ContourMeta.pb`
- component-to-family mapping through `component_name -> family`

Relevant files:

- `proto/NPWiseVoxelData.proto`
- `proto/NPInputVoxelData.proto`
- `proto/generate.bat`

### DICOM inputs

Supported DICOM path today:

- `RT Dose .dcm`
- optional `RTSTRUCT .dcm`

Current assumptions:

- only absolute `Gy` RT Dose inputs are supported
- current DICOM path is single-field only
- the structure is selected by ROI name through `Structure`

### NIfTI contour input

Supported path:

- one aligned binary mask per target structure

Current assumptions:

- mask already matches the target voxel grid shape
- multi-label segmentations are not supported

### Folder autodiscovery

The pipeline accepts a directory instead of explicit files.

Folder mode is intended for one-animal-per-folder layouts.

Autodiscovery rules:

- for DICOM mode, it expects one `RT Dose` and optionally one `RTSTRUCT`
- for protobuf mode, it looks for one dose input and one geometry input
- if multiple candidates are found, the pipeline raises an explicit error and requires manual selection

The autodiscovery logic lives in `resolve_pipeline_input_paths()` inside
`pipeline_geant4_to_prediction.py`.

## Key Public APIs By File

### `fit_alpha_beta_using_processor.py`

Primary entry points:

- `Fitter`
- `Fitter.fit_let_dependence()`
- `main()`

Responsibilities:

- fit radiobiological model families
- derive LET-dependent fits from family results
- provide analysis summaries and CSV export

### `dose_reader.py`

Primary types:

- `VoxelDose`
- `DoseMap`

Primary functions:

- `read_dose_map()`
- `read_full_dose_map()`
- `is_rt_dose_path()`
- `is_rtstruct_path()`

Responsibilities:

- deserialize protobuf GEANT4 maps or RT Dose DICOM
- attach geometry and structure membership
- expose helper methods for tumor voxel selection, DVH, mean dose, and mean LET

### `rtstruct_bridge.py`

Primary types:

- `RTDoseGridGeometry`
- `StructureAssignments`

Primary function:

- `build_structure_assignments_from_rtstruct()`

Responsibilities:

- read ROI names from RTSTRUCT
- select ROI by requested structure name
- rasterize planar contours onto the RT Dose grid

### `nifti_contour_bridge.py`

Primary functions:

- `build_structure_assignments_from_nifti()`
- `build_structure_assignments_from_nifti_grid()`

Responsibilities:

- convert one aligned NIfTI mask into `structure_ids` and `voxel_structure_ids`

### `let_parametrization.py`

Primary type:

- `LETDependentParams`

Primary function:

- `fit_let_dependence()`

Responsibilities:

- fit or carry a linear LET-dependent parameterization for `alpha` and `beta`

### `voxel_sf_calculator.py`

Primary types:

- `VoxelSF`
- `VolumetricSFResult`

Primary functions:

- `compute_voxel_sf()`
- `compute_mixed_voxel_sf()`

Responsibilities:

- evaluate voxel-level `SF` and `BED`
- aggregate to structure-level summaries such as `mean_sf`, `mean_dose_gy`, `D90`, `EUD`, `effective_alpha`, `effective_beta`

### `mixed_field_model.py`

Primary types:

- `FieldComponent`
- `MixedFieldResult`

Primary function:

- `compute_mixed_field_sf()`

Responsibilities:

- combine family-specific field components into a structure-level mixed-field estimate

### `tumor_growth_predictor.py`

Primary types:

- `TreatmentFraction`
- `GeometryReference`
- `GrowthModelParameters`
- `GrowthSimulationResult`

Primary functions:

- `simulate_growth()`
- `fit_gompertz_to_control()`
- `predict_schedule_surviving_fraction()`

Responsibilities:

- simulate live/dead volume evolution
- support repair-aware schedules
- optionally override scalar `alpha/beta` inputs with a `volumetric_sf` result from the voxel pipeline

### `radiobiology_analysis.py`

Primary functions:

- `compute_rbe()`
- `compute_rbe_let()`
- `compute_tcp()`
- `build_tcp_curve()`
- `compute_ntcp_lkb()`
- `build_ntcp_curve()`
- `export_bed_eqd2_table()`
- sensitivity and scenario comparison helpers

Responsibilities:

- secondary analyses and reporting on fitted or predicted radiobiological behavior

### `pipeline_geant4_to_prediction.py`

Primary functions:

- `run_prediction_pipeline()`
- `resolve_pipeline_input_paths()`
- `build_cli_parser()`
- `cli_main()`

Responsibilities:

- orchestrate dose ingest, radiobiology resolution, voxel aggregation, growth simulation, and export

### `geant4_pipeline_gui.py`

Primary types/functions:

- `PipelineRadiobiologySource`
- `resolve_radiobiology_source_from_run_results()`
- `Geant4PipelineWindow`

Responsibilities:

- GUI layer for file or folder selection, radiobiology source resolution, and pipeline execution

## Output Artifacts

The pipeline writes the following standard files:

- `dvh_<structure>.csv`
- `sf_per_voxel.csv`
- `aggregated_params.json`
- `growth_curve.csv`
- `bed_eqd2_table.csv`
- `summary.json`

The CLI also prints a short console summary:

- `mean_dose_gy`
- `mean_sf`
- `effective_alpha`
- `effective_beta`
- `summary_json`

## Test Coverage

Module-local tests in `survival/tests/`:

- `test_dose_reader.py`
  - protobuf, RT Dose, NIfTI, and RTSTRUCT ingest
- `test_pipeline.py`
  - CLI parsing, pipeline output generation, RT Dose mode, RTSTRUCT mode, folder autodiscovery
- `test_voxel_sf_calculator.py`
  - voxel-level and volumetric `SF/BED` calculations
- `test_mixed_field_model.py`
  - mixed-field composition logic
- `test_let_parametrization.py`
  - LET parameter fitting
- `test_geant4_pipeline_gui.py`
  - GUI source resolution and pipeline-window helpers

Broader regression coverage also exists in the repository-level test suite for:

- fitter backend
- fitter GUI
- radiobiology analysis helpers
- tumor growth predictor
- predictor GUI

## Current Limitations

- `RT Dose` support is currently single-field only.
- `fullVoxelMap` mixed-field logic remains protobuf-only.
- NIfTI support expects one binary mask already aligned to the target grid.
- Multi-label NIfTI segmentations are not supported.
- Folder mode expects one case per directory and does not try to resolve ambiguous multi-case folders automatically.
- The plan documents are historical and may intentionally lag behind the current implementation state.
