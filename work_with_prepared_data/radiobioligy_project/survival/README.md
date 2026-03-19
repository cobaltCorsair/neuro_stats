# `fit_alpha_beta_using_processor.py`

This module fits effective radiobiological parameters from in vivo tumor-volume Excel files.
It is intended for series where:

- control curves are available for normalization;
- dose fractions can be parsed from experiment metadata;
- response is measured through tumor-volume dynamics rather than a clonogenic assay.

The fitter supports both CLI and GUI workflows.

For a full Russian-language user manual with GUI workflows and common use cases, see [USER_GUIDE_RU.md](USER_GUIDE_RU.md).

## What It Does

- reads `.xlsx` tumor-volume files;
- builds pooled or explicitly assigned control curves;
- normalizes treated curves to control;
- computes scalar `SF` endpoints or uses the full normalized response curve;
- fits multiple model families:
  - `classic_lq`
  - `repair_lq`
  - `lq_l`
  - `glq`
  - `lq_repop`
  - `repair_repop`
  - `linear`
- supports train/validation splits by regimen kind;
- supports bootstrap over treated and control animals;
- supports repeated-regimen aggregation and deduplication;
- exports batch summaries to CSV;
- exposes structured results for the GUI and tests.

## Response Definition

Supported scalar `SF` modes:

- `absolute`
  - `SF = min(mean_norm[1:]) / mean_norm[0]`
- `absindex:N`
  - `SF = mean_norm[N] / mean_norm[0]`

Multiple scalar endpoints can be evaluated in one run:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor ^
  --files *.xlsx ^
  --sf absolute ^
  --sf absindex:1 ^
  --sf absindex:2
```

## Response Modes

Two response modes are available:

- `scalar`
  - fits one `SF` value per regimen.
- `curve`
  - fits the full normalized tumor-response curve;
  - additionally estimates `curve_clearance_rate` to describe late-time recovery.

CLI examples:

```bash
--response-mode scalar
--response-mode curve
```

Note:
- in `curve` mode only the first `--sf` entry is used, because the remaining `SF` modes belong to the scalar setup.

## Model Families

Available model selections:

- `--model-kind auto`
- `--model-kind classic_lq`
- `--model-kind repair_lq`
- `--model-kind lq_l`
- `--model-kind glq`
- `--model-kind lq_repop`
- `--model-kind repair_repop`
- `--model-kind linear`

Model meaning:

- `classic_lq`
  - `SF = exp(-alpha * D - beta * sum(d_i^2))`
- `repair_lq`
  - classic LQ with explicit `t=` timing metadata and repair via `--repair-half-time-hours`
- `lq_l`
  - LQ-L model with a fitted `transition_dose`
- `glq`
  - generalized high-dose LQ variant with fitted `saturation_dose`
- `lq_repop`
  - LQ plus delayed repopulation with fitted `lag_days` and `repopulation_rate`
- `repair_repop`
  - repair-aware LQ plus delayed repopulation
- `linear`
  - no quadratic term, effectively `beta = 0`

To compare models on the same train set:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor ^
  --files *.xlsx ^
  --family y ^
  --response-mode curve ^
  --repair-half-time-hours 1.0 ^
  --compare-models
```

Model comparison ranks candidates by train-set quality using:

- `AIC`
- `RMSE`
- `mean_abs_log_error`
- `MAE`

## Timing And Repair

If experiment metadata contains `t=...`, the fitter can recover real inter-fraction timing in hours, half-hours, days, and similar intervals.
If metadata also contains irradiation-duration markers such as `tau`, `t_irr`, or `duration`, the repair-aware quadratic term now treats each fraction as a finite exposure instead of an instantaneous pulse.

Examples:

- `t = 30 min`
- `t = 1 hr`
- `t = 2.5 hr`
- `t = 1 day`

Repair-aware fitting is enabled by:

```bash
--repair-half-time-hours 1.0
```

If no explicit timing metadata is available, the fitter falls back to schedule-free behavior.

## Family Detection

Current family mapping:

- `y` -> gamma / photon series
- `p` -> proton series without explicit beam-position context
- `p_peak` -> proton peak series (`in_peak`, corresponding peak markers in file names)
- `p_through` -> proton shoot-through / pass-through series
- `n` -> neutron series
- `e` -> electron series
- `c` -> carbon-ion C-12 series

Examples:

- `22.10.2025_p40_in_peak.xlsx` -> `p_peak`
- `08.10.2021_p_32_...xlsx` -> `p_through`
- `05.12.2018_c_12.xlsx` -> `c`

Different families should not be mixed into one fit unless you have a clear radiobiological justification.

## Train / Validation Splits

Supported regimen kinds:

- `all`
- `single`
- `fractionated`

Typical workflow:

1. fit on `single`;
2. validate on `fractionated`.

Example:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor ^
  --files *.xlsx ^
  --family y ^
  --fit-kind single ^
  --validate-kind fractionated ^
  --sf absolute
```

## Bootstrap

Enable bootstrap with:

```bash
--bootstrap N
```

The bootstrap resamples:

- animals inside treated groups;
- animals inside control groups.

Output includes:

- mean
- std
- median
- `95%` interval

Example:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor ^
  --files *.xlsx ^
  --family y ^
  --sf absolute ^
  --bootstrap 500 ^
  --bootstrap-seed 7
```

## Repeated Regimens

If one family contains multiple files with the same regimen, two tools are available.

Aggregate repeated regimens:

```bash
--aggregate-regimens
```

This collapses repeats into one weighted regimen with:

- mean `SF`
- repeat count
- `sf_std`

Deduplicate repeated regimens:

```bash
--dedupe-regimens
```

This keeps only one file per `(family, fractions, schedule)` key.

## Batch Mode

To analyze all detected families separately:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor ^
  --files *.xlsx ^
  --by-family ^
  --fit-kind single ^
  --validate-kind fractionated ^
  --sf absolute ^
  --aggregate-regimens ^
  --bootstrap 200
```

## Summary CSV

To export a batch summary:

```bash
--summary-csv C:\dev\neuro_stats\family_summary.csv
```

The CSV contains:

- `sf_mode`
- `response_mode`
- `model_kind`
- `family`
- `status`
- `total_count`
- `single_count`
- `fractionated_count`
- `train_count`
- `validation_count`
- `alpha`
- `beta`
- `alpha_beta_ratio`
- `reason`

## Secondary Analyses

The module [radiobiology_analysis.py](/C:/dev/neuro_stats/work_with_prepared_data/radiobioligy_project/survival/radiobiology_analysis.py) adds higher-level radiobiological analyses on top of fitted runs and predictor trajectories.

Current capabilities:

- `RBE` estimation relative to a reference family such as `y`
- iso-effect dose conversion for single-fraction comparisons
- batch `RBE` series for `2 Gy`, `10 Gy`, or any custom dose grid
- one-at-a-time sensitivity analysis for:
  - `alpha`
  - `beta`
  - `growth_rate`
  - `carrying_capacity`
  - `clearance_rate`
  - physical dose
- interval sensitivity for fraction gaps such as `0.5 h`, `1 h`, `2.5 h`, `24 h`
- cross-comparison of scalar `SF` definitions like `absolute`, `absindex:1`, `absindex:2`

Typical imports:

```python
from work_with_prepared_data.radiobioligy_project.survival.radiobiology_analysis import (
    analyze_interval_sensitivity,
    analyze_parameter_sensitivity,
    build_rbe_series,
    compare_sf_metric_sensitivity,
    compute_rbe,
)
```

`RBE` is defined as:

```text
RBE = D_ref / D_test
```

for the same predicted biological effect, with `y` usually used as the reference family.

Important interpretation notes:

- `RBE` here is derived from the fitted effective `alpha/beta` response, not from a separate microdosimetric transport model.
- sensitivity analysis uses predictor `total-volume RMSE` as the score.
- interval sensitivity is most informative when timing can actually change the predicted curve, for example via repair, clearance, growth, or sub-day observations.

## GUI

Launch:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_gui
```

### Current Layout

The fitter window now uses a compact two-column layout:

- left sidebar
  - action buttons
  - `Files` tab
  - `Fit setup` tab
- right result area
  - top `Run summary`
  - bottom detail tabs

This layout is meant to fit better on a normal laptop screen than the older long vertical window.

### What The GUI Supports

- drag-and-drop `.xlsx` files;
- adding files or a whole folder;
- explicit `experiment -> control` mapping when several control files are loaded;
- `Default control` plus `Apply to all`;
- setup of `SF modes`, `family`, split strategy, response mode, model kind, repair half-time, bootstrap, and CSV export;
- inventory scan before fitting;
- run summary table plus per-run detail tabs;
  the top summary now includes mean train-set `BED`, `EQD2`, and `G` alongside fit quality metrics;
- `Analysis` tab with:
  - `RBE mode`: either classic per-family iso-effect comparison or `LET model` based RBE from one `let_dependent` fit
  - `RBE vs dose`
  - `RBE vs alpha/beta`
  - `alpha(LET)` line for `let_dependent` fits, with per-family alpha reference points when independent family fits are available
  - `SF`-metric drift table for the current response/model context;
  - `Export CSV`, which writes paired analysis files such as `..._rbe.csv`, `..._sf_metrics.csv`, and `..._alpha_let.csv` when LET analysis is available;
- `NTCP` tab with both a manual LKB curve builder and automatic `TD50/m` fitting from skin/RTOG Excel groups (`peak RTOG >= threshold`);
- launch of the tumor growth predictor window.

### Inventory Mode

The `Inventory` tab shows file-by-file:

- `family`
- `kind`
- `fractions`
- `schedule`
- `control`
- `fit ready`
- `notes`

The family summary also reports heuristic model targets:

- `recommended`
- `possible`

for:

- `classic_lq`
- `repair_lq`
- `lq_l`
- `glq`
- `lq_repop`
- `repair_repop`

These heuristics look at:

- dose contrast;
- explicit `t=` timing;
- high-dose coverage;
- follow-up curve length.

### Typical GUI Workflow

1. Load control and treated `.xlsx` files.
2. If several controls are loaded, fill the experiment-to-control mapping.
3. Choose family and response mode.
4. If needed, set `fit-kind = single` and `validate-kind = fractionated`.
5. If timing matters, provide a positive `Repair T1/2 (h)`.
6. Run the fit.
7. Review the summary table and the selected run details.

## Main CLI Examples

Simple fit in the current folder:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor
```

Family-specific fit:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor ^
  --files *.xlsx ^
  --family y ^
  --sf absolute
```

Full-curve fit:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor ^
  --files *.xlsx ^
  --family y ^
  --response-mode curve ^
  --model-kind classic_lq
```

Repair-aware comparison:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor ^
  --files *.xlsx ^
  --family y ^
  --response-mode scalar ^
  --repair-half-time-hours 1.0 ^
  --compare-models
```

Explicitly disable LOO cross-validation:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor ^
  --files *.xlsx ^
  --no-cross-validate-loo
```

## Current Limitations

- these are effective in vivo parameters derived from tumor-volume dynamics, not clonogenic `alpha/beta`;
- repair-aware fitting requires explicit `t=` timing metadata in the Excel file;
- control assignment in the GUI is explicit, but CLI still pools controls unless a control map is passed through the API;
- high-dose model comparison becomes meaningful only when the dataset truly covers a broad high-dose range;
- sparse datasets can fit mathematically but still be weak biologically.

## Quick CLI Help

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor --help
```
