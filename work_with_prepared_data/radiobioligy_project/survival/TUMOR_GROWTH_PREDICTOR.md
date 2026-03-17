# Tumor Growth Predictor

`tumor_growth_predictor_gui.py` is a separate GUI window for forward simulation of tumor growth and treatment response.

It combines:

- fitted `alpha` and `beta`
- an editable irradiation schedule
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
- a time slider
- `Play/Pause`
- a text summary of the current model state

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

- `Day`
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

## Related Files

- `fit_alpha_beta_gui.py`: survival fitter GUI
- `fit_alpha_beta_using_processor.py`: fitter backend
- `tumor_growth_predictor.py`: mathematical model
- `tumor_growth_predictor_gui.py`: predictor window
- `tumor_geometry_processor.py`: parser for `a-b-c`
- `tumor_3d_viewer.py`: separate 3D viewer for observed geometry only
