# Tumor 3D Viewer

`tumor_3d_viewer.py` is a standalone PyQt6 window for visualizing tumor geometry over time when the Excel file contains measured orthogonal axes in `a-b-c` format.

## What It Shows

- A 3D ellipsoid reconstructed from the measured diameters `a`, `b`, `c`
- A day slider for stepping through the experiment
- Smooth `Play/Pause` animation over time
- Selection of either one rat or the mean geometry across rats
- `Wireframe` and `Surface` render modes
- A per-day table with `day`, `a`, `b`, `c`, `volume`, and the geometry source

## Important Limitation

This viewer does **not** reconstruct the true anatomical tumor shape.
It renders an ellipsoid consistent with the measured diameters:

- If a cell contains `a-b-c`, the displayed shape uses those three axes directly.
- If a cell contains only a scalar volume, the viewer falls back to an equivalent sphere with the same volume.

So this is a geometry-preserving visualization of caliper measurements, not a tomographic 3D segmentation.

## Input Format

The viewer expects the same Excel layout used by the tumor-processing scripts:

- Row 1: experiment parameters
- Row 2: day labels
- Column 1: rat labels
- Remaining cells: either `a-b-c` or a scalar tumor volume

Examples of valid cells:

- `2-4-6`
- `3.5-3.0-2.5`
- `8.2`

## Launch

From the repository root:

```powershell
python -m work_with_prepared_data.radiobioligy_project.tumor_3d_viewer
```

If you use Poetry:

```powershell
poetry run python -m work_with_prepared_data.radiobioligy_project.tumor_3d_viewer
```

## Usage

1. Open an `.xlsx` file or drag it into the window.
2. Choose `Mean across rats` or one rat in the `Tumor` selector.
3. Move the slider to inspect geometry on a specific day.
4. Switch between `Wireframe` and `Surface` rendering if needed.
5. Use the table on the right to jump directly to a day.

The viewer interpolates geometry between neighboring measurement days during playback and fine slider moves.
This gives a smoother visual transition while keeping the original discrete measurements in the table.

## Data Source Semantics

The `Source` column in the right-hand table means:

- `axes`: this day has an explicit `a-b-c` measurement
- `sphere`: this day had only a scalar volume, so the viewer displays an equivalent sphere
- `X/Y axes`: in mean mode, `X` rats out of `Y` had explicit `a-b-c` measurements on that day

## Related Files

- `data_processing/tumor_geometry_processor.py`: Excel parser that preserves `a-b-c`
- `tumor_3d_viewer.py`: standalone GUI window
- `data_processing/excel_data_processor.py`: legacy scalar-volume parser used by older analysis paths
