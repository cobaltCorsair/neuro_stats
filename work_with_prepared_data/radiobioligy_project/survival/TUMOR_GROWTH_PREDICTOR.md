# Tumor Growth Predictor

`tumor_growth_predictor_gui.py` is a separate GUI window for forward simulation of tumor growth and treatment response.

It combines:

- fitted `alpha` and `beta`
- an editable irradiation schedule
- optional repair-aware interaction between closely spaced fractions
- Gompertz growth between fractions
- delayed clearance of damaged tumor mass
- reconstruction of a 3D ellipsoid from measured `a-b-c`

This document explains what files to load, what each control means, and how to interpret the output.

## What Problem It Solves

The survival fitter estimates effective `alpha` and `beta` from tumor-volume response.
The growth predictor uses those parameters to answer a different question:

- given a treatment schedule,
- given a baseline tumor geometry,
- given a control-derived growth trend,

what tumor volume and what ellipsoid axes should we expect over time?

So this window is for **forward prediction**, not for fitting `alpha/beta`.

## Which Files You Need

The predictor works with two Excel inputs.

### 1. Treated tumor file

This is the main experimental `.xlsx` file for the irradiated series you want to simulate.

It is expected to contain:

- experiment parameters in the first row
- day labels in the second row
- rat labels in the first column
- tumor measurements in the remaining cells

Cells may contain:

- `a-b-c`
- or a scalar volume

Examples:

- `11.04.2025_y4_y4_y32.xlsx`
- `19.03.2025_y_40.xlsx`

The predictor uses this file for:

- the observed days
- the observed tumor volume trajectory
- the observed `a`, `b`, `c` trajectory
- the baseline ellipsoid
- optional auto-prefill of the dose schedule from the experiment header

If the first row contains irradiation metadata such as `t = 1 ч`, the predictor
also uses it to reconstruct the spacing between fractions. Internally the model
still works in days, so `1 ч` becomes `1/24` day.

### 2. Control file

This is a non-irradiated control series.

Examples:

- `control_2016.xlsx`

The predictor uses the control file only to estimate untreated growth, currently via a Gompertz fit.

If you do not load a control file, you can still enter growth parameters manually.

## Window Layout

The predictor window has three main parts.

### Left panel

This is where you define the model input.

It contains:

- `Treated tumor`: the irradiated tumor file you want to model
- `Control`: the control file used for untreated growth fitting
- `Tumor`: selection of either one rat or `Mean across rats`
- `Alpha/Beta source`: either manual values or one of the recent fitter results
- model parameters
- editable dose schedule

### Upper-right panel

This panel shows 2D curves:

- predicted total volume
- predicted live volume
- predicted dead volume
- observed volume points
- predicted `a`, `b`, `c`
- observed `a`, `b`, `c`

### Lower-right panel

This panel shows:

- the predicted 3D ellipsoid
- a day-by-day slider
- a `Frames` switch with `Daily snapshots` and `Raw timeline`
- a `Speed` switch for faster playback, especially in `Raw timeline`
- `Play/Pause`
- a text summary of the current model state

The underlying simulation can still run on a finer sub-day grid when hour-scale
fractions are present, but the 3D tab intentionally displays daily snapshots so
the animation stays readable.
If you need to inspect the exact sub-day model states, switch `Frames` to
`Raw timeline`.

## Mathematical Model

The predictor uses two compartments:

- `V_live`: viable proliferating tumor volume
- `V_dead`: radiation-damaged volume that clears gradually

### Between fractions

Live tumor growth follows a Gompertz law:

```text
dV_live/dt = r * V_live * ln(K / V_live)
```

Damaged volume clears exponentially:

```text
dV_dead/dt = -k_clear * V_dead
```

### At each irradiation event

For one fraction dose `d`, the surviving fraction is:

```text
SF = exp(-alpha * d - beta * d^2)
```

Then:

```text
V_live(t+) = SF * V_live(t-)
V_dead(t+) = V_dead(t-) + (1 - SF) * V_live(t-)
```

The plotted total volume is:

```text
V_total = V_live + V_dead
```

### Repair-aware mode for short intervals

If `Repair T1/2 (h)` is greater than zero, the predictor keeps a decaying memory
of unrepaired sublethal damage between fractions.

Conceptually:

```text
unrepaired(t + dt) = unrepaired(t) * exp(-mu * dt)
mu = ln(2) * 24 / T1/2_hours
```

At the next fraction dose `d`, the quadratic LQ term becomes:

```text
d^2 + 2 * d * unrepaired_before_fraction
```

This means:

- `T1/2 = 0` keeps the classic predictor behavior
- `T1/2 > 0` makes `30 min`, `1 h`, `2.5 h` and similar gaps matter radiobiologically
- short gaps increase effective kill compared with the same doses delivered far apart

## Geometry Reconstruction Modes

The predictor currently supports two geometry modes.

### Fixed ratios

This is the conservative mode.

It assumes the ellipsoid keeps the same axis proportions and only scales with total volume:

```text
a(t) = a0 * (V_total(t) / V0)^(1/3)
b(t) = b0 * (V_total(t) / V0)^(1/3)
c(t) = c0 * (V_total(t) / V0)^(1/3)
```

Use this when:

- you want a stable default
- observed shape data are noisy
- you only trust volume more than separate axes

### Fit from observed shape

This mode fits each axis independently from the observed treated-tumor trajectory:

```text
a(t) = c_a * V_total(t)^p_a
b(t) = c_b * V_total(t)^p_b
c(t) = c_c * V_total(t)^p_c
```

This means the model can learn that one axis changes faster or slower than the others.

Use this when:

- your treated file has enough reliable `a-b-c` points
- you want shape evolution, not only isotropic scaling

If the observed data are too weak for fitting, the code falls back to the fixed-ratio model.

## Recommended Workflow

### Fast start

1. Open the survival fitter GUI:

```powershell
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_gui
```

2. Run your usual `alpha/beta` analysis.
3. Click `Open growth predictor`.
4. In the predictor window, load the treated tumor file.
5. Load the control file.
6. Choose the desired tumor: one rat or `Mean across rats`.
7. Select an `Alpha/Beta source` from the fitted runs, or leave `Manual`.
8. Click `Fit Gompertz from control`.
9. Check the schedule table and edit days or doses if needed.
10. Choose `Geometry mode`.
11. Click `Simulate`.
12. Inspect curves and the 3D ellipsoid over time.

If the schedule was auto-filled from a header like `t = 1 ч`, the first column
shows time in days, for example:

- `0`
- `0.0417`
- `0.0833`

which correspond to `0 h`, `1 h`, and `2 h`.

### Standalone start

You can also open the predictor directly:

```powershell
python -m work_with_prepared_data.radiobioligy_project.survival.tumor_growth_predictor_gui
```

In that case, `alpha` and `beta` start as manual values until you enter them.

## Meaning of the Controls

### `Alpha/Beta source`

- `Manual`: use the values currently written in the spin boxes
- fitted run entry: copy `alpha` and `beta` from a previous fitter result

### `Growth rate r`

Controls how fast the viable compartment grows between fractions.

Higher `r` means faster regrowth.

### `Repair T1/2 (h)`

Controls how quickly sublethal radiation damage repairs between fractions.

- `0`: classic predictor, each fraction uses only its own `d` and `d^2`
- positive value: repair-aware predictor, close fractions interact through incomplete repair

Examples:

- `0.5` means fast repair, so interaction fades within a few hours
- `1.0` means strong sensitivity to hour-scale spacing
- `4.0` means interaction persists longer between fractions

### `Dose schedule`

The schedule table uses `Time (days)` rather than integer treatment days.

This is important for experiments where fractions are separated by hours rather
than by full days. When the treated file contains a header token starting with
`t =`, the predictor tries to parse that interval automatically.

Examples:

- `t = 1 ч` -> repeated 1-hour gaps between fractions
- `t = 24 ч` -> repeated 1-day gaps between fractions

If no valid `t =` token is found, the auto-prefill falls back to `0, 1, 2, ...`
days.

If `Repair T1/2 (h)` is enabled, those sub-day gaps affect not only the plot
timeline but also the radiation kill itself.

### `Carrying capacity K`

Upper growth scale for the Gompertz model.

Higher `K` allows larger untreated tumor size.

### `Clearance rate`

Controls how quickly damaged tumor mass disappears.

Higher values mean the model removes `V_dead` faster, so post-treatment shrinkage can appear earlier and stronger.

### `Horizon (days)`

The maximum simulated day.

### `Step (days)`

Time resolution of the prediction grid.

Smaller values give smoother curves and animation, but produce more points.

### `Dose schedule`

Each row is one irradiation event:

- `Time (days)`
- `Dose (Gy)`

The schedule can be prefilled from the treated file header and then edited manually.

## How To Read the Output

### Volume plot

- `Predicted total`: what the model says should be observed experimentally
- `Predicted live`: viable tumor compartment
- `Predicted dead`: damaged compartment still present in the lesion
- `Observed volume`: real points from the treated file

### Axis plot

- lines: predicted `a`, `b`, `c`
- markers: observed `a`, `b`, `c`

This is the most useful panel when you want to check whether the model gets shape dynamics right.

### 3D panel

The 3D view shows the predicted ellipsoid at the selected day.

Important:

- this is not tomographic reconstruction
- it is a model ellipsoid consistent with the predicted axes

### Summary panel

The summary reports:

- current files
- current `alpha`, `beta`
- growth parameters
- repair setting
- geometry mode
- current predicted `V_total`, `V_live`, `V_dead`
- current `a`, `b`, `c`
- reference geometry
- geometry scaling law
- dose schedule

## Practical Advice

### If you have a good control

Use `Fit Gompertz from control`.
That is usually better than guessing `r` and `K`.

### If shape is noisy

Start with `Fixed ratios`.
Then compare against `Fit from observed shape`.

### If prediction overshoots volume

Possible causes:

- `growth_rate` too high
- `carrying_capacity` too high
- `alpha/beta` too weak
- schedule entered incorrectly

### If post-treatment shrinkage is too slow

Possible causes:

- `clearance_rate` too low
- `alpha/beta` too weak

### If the 3D ellipsoid looks unrealistic

Check:

- baseline `a-b-c`
- geometry mode
- whether observed axis data are sparse or noisy

## What This Model Is Not

This is a **phenomenological in vivo model**.

It is useful for:

- exploring treatment scenarios
- comparing schedules
- visualizing expected dynamics
- generating hypothesis-level predictions

It is not yet:

- a mechanistic tissue model
- a clonogenic gold-standard radiobiology model
- a segmentation-based anatomical 3D reconstruction

Current `alpha/beta` values in this project are effective parameters derived from tumor-volume response, not classical clonogenic constants.
Repair-aware mode also uses a single exponential repair half-time, not a full multicomponent repair model.

## Related Files

- `fit_alpha_beta_gui.py`: survival fitter GUI
- `fit_alpha_beta_using_processor.py`: fitter backend
- `tumor_growth_predictor.py`: mathematical model
- `tumor_growth_predictor_gui.py`: predictor window
- `tumor_geometry_processor.py`: parser for `a-b-c`
- `tumor_3d_viewer.py`: separate 3D viewer for observed geometry only
