"""Summarise the literature-bounded ROKUS gamma source surrogate.

The model combines a finite 20-mm source face, a virtual aperture at the
published ROKUS shutter position and an approximate published ROKUS-M photon
spectrum.  It is a sensitivity model, not commissioning of a specific unit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


NEURO_STATS_ROOT = Path(__file__).resolve().parents[3]
if str(NEURO_STATS_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURO_STATS_ROOT))

from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (  # noqa: E402
    MatplotlibConfigurator,
)


def read_single_csv(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {path}, found {len(rows)}")
    return rows[0]


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_dvh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["resolution"] == "report_1p6x1p6x0p8"
        ]
    return (
        np.asarray([float(row["dose_over_gtv_mean"]) for row in rows]),
        np.asarray(
            [
                float(row["volume_receiving_at_least_dose_percent"])
                for row in rows
            ]
        ),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def projected_profile(
    field_mm: float,
    *,
    histories: int = 5_000_000,
    seed: int = 20260729,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Monte-Carlo fluence profile in the isocentre plane.

    The profile averages a 2-mm-wide strip around the central cross-axis.  It
    is independent of Geant4 particle transport and tests source/aperture
    geometry only.
    """

    source_to_isocentre_mm = 750.0
    source_to_aperture_mm = 274.0
    source_radius_mm = 10.0
    scale = source_to_isocentre_mm / source_to_aperture_mm
    aperture_half_mm = (
        0.5
        * field_mm
        * source_to_aperture_mm
        / source_to_isocentre_mm
    )
    edges = np.linspace(-60.0, 60.0, 481)
    histogram = np.zeros((480, 480), dtype=np.int64)
    rng = np.random.default_rng(seed + int(field_mm))
    remaining = histories
    while remaining:
        count = min(500_000, remaining)
        radius = source_radius_mm * np.sqrt(rng.random(count))
        angle = 2.0 * np.pi * rng.random(count)
        source_x = radius * np.cos(angle)
        source_z = radius * np.sin(angle)
        aperture_x = rng.uniform(-aperture_half_mm, aperture_half_mm, count)
        aperture_z = rng.uniform(-aperture_half_mm, aperture_half_mm, count)
        x = source_x + (aperture_x - source_x) * scale
        z = source_z + (aperture_z - source_z) * scale
        block, _, _ = np.histogram2d(x, z, bins=(edges, edges))
        histogram += block.astype(np.int64)
        remaining -= count

    centres = 0.5 * (edges[:-1] + edges[1:])
    profile = histogram[:, np.abs(centres) <= 1.0].mean(axis=1)
    profile = profile.astype(float)
    profile /= profile[np.abs(centres) <= 2.0].mean()
    profile = np.convolve(profile, np.ones(5) / 5.0, mode="same")

    positive = centres >= 0.0
    x_positive = centres[positive]
    y_positive = profile[positive]
    peak_index = int(np.argmax(y_positive))

    def crossing(level: float) -> float:
        indices = np.flatnonzero(y_positive <= level)
        indices = indices[indices > peak_index]
        index = int(indices[0])
        x0, x1 = x_positive[index - 1 : index + 1]
        y0, y1 = y_positive[index - 1 : index + 1]
        return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))

    x80 = crossing(0.8)
    x50 = crossing(0.5)
    x20 = crossing(0.2)
    return centres, profile, {
        "nominal_field_mm": field_mm,
        "x80_mm": x80,
        "x50_mm": x50,
        "x20_mm": x20,
        "penumbra_80_20_mm": x20 - x80,
        "full_width_at_50_percent_mm": 2.0 * x50,
    }


def single_run_row(
    label: str,
    path: Path,
    status: str,
) -> dict[str, object]:
    row = read_single_csv(path)
    return {
        "scenario": label,
        "status": status,
        "histories": int(row["histories"]),
        "gtv_mean_dose_Gy_per_accepted_photon": float(
            row["gtv_mean_dose_Gy_per_primary"]
        ),
        "gtv_LETd_w_keV_um": float(row["gtv_LETd_w_keV_um"]),
        "report_D50_over_Dmean": float(
            row["gtv_aggregated_D50_over_Dmean_1p6x1p6x0p8"]
        ),
        "report_D90_over_Dmean": float(
            row["gtv_aggregated_D90_over_Dmean_1p6x1p6x0p8"]
        ),
        "fine_nonzero_dose_fraction": float(
            row["gtv_nonzero_dose_fraction"]
        ),
    }


def pooled_row(
    label: str,
    path: Path,
    status: str,
) -> dict[str, object]:
    row = read_json(path)
    histories = int(row["pooled_histories"])
    return {
        "scenario": f"{label}, {histories / 1_000_000:g} млн",
        "status": status,
        "histories": histories,
        "gtv_mean_dose_Gy_per_accepted_photon": float(
            row["pooled_gtv_mean_dose_Gy_per_primary"]
        ),
        "gtv_LETd_w_keV_um": float(row["pooled_gtv_LETd_w_keV_um"]),
        "report_D50_over_Dmean": float(
            row["pooled_report_D50_over_Dmean_1p6x1p6x0p8"]
        ),
        "report_D90_over_Dmean": float(
            row["pooled_report_D90_over_Dmean_1p6x1p6x0p8"]
        ),
        "fine_nonzero_dose_fraction": float(
            row["pooled_gtv_nonzero_dose_fraction"]
        ),
    }


def analyse(args: argparse.Namespace) -> list[dict[str, object]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        pooled_row(
            "параллельное поле, две линии, pooled",
            args.baseline_summary,
            "reference entrance-plane source",
        ),
        single_run_row(
            "геометрия РОКУС, две линии, 0,5 млн",
            args.geometry_only_summary,
            "finite source and virtual aperture only",
        ),
        single_run_row(
            "спектр РОКУС, точечный источник, 0,5 млн",
            args.spectrum_only_summary,
            "literature spectrum only",
        ),
        pooled_row(
            "РОКУС 30×30 мм, pooled",
            args.rokus_30_summary,
            "main literature-bounded surrogate",
        ),
        single_run_row(
            "РОКУС 40×40 мм, 1 млн",
            args.rokus_40_summary,
            "field-size sensitivity",
        ),
    ]
    write_csv(args.output_dir / "gamma_rokus_surrogate_summary.csv", rows)

    spectrum_rows: list[dict[str, str]]
    with args.spectrum_csv.open(encoding="utf-8-sig", newline="") as handle:
        spectrum_rows = list(csv.DictReader(handle))
    energies = np.asarray(
        [float(row["energy_MeV"]) for row in spectrum_rows]
    )
    probabilities = np.asarray(
        [float(row["probability"]) for row in spectrum_rows]
    )
    components = np.asarray([row["component"] for row in spectrum_rows])

    profiles: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    penumbra_rows: list[dict[str, float]] = []
    profile_rows: list[dict[str, float]] = []
    for field in (30.0, 40.0):
        x, profile, metrics = projected_profile(field)
        profiles[field] = (x, profile)
        penumbra_rows.append(metrics)
        for coordinate, fluence in zip(x, profile):
            profile_rows.append(
                {
                    "nominal_field_mm": field,
                    "cross_axis_mm": coordinate,
                    "relative_fluence": fluence,
                }
            )
    write_csv(args.output_dir / "gamma_rokus_geometric_penumbra.csv", penumbra_rows)
    write_csv(args.output_dir / "gamma_rokus_projected_profiles.csv", profile_rows)

    baseline_x, baseline_y = read_dvh(args.baseline_dvh)
    rokus_x, rokus_y = read_dvh(args.rokus_30_dvh)

    MatplotlibConfigurator().apply_custom_styles()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

    ax = axes[0, 0]
    line_mask = components == "Co60_line"
    ax.bar(
        energies[~line_mask],
        probabilities[~line_mask],
        width=0.035,
        color="#80b1d3",
        label="оцифрованный континуум",
    )
    ax.bar(
        energies[line_mask],
        probabilities[line_mask],
        width=0.025,
        color="#fb8072",
        label="линии Co-60",
    )
    ax.set(
        title="Аппроксимация спектра РОКУС-М; средняя 1,058 МэВ",
        xlabel="Энергия фотона, МэВ",
        ylabel="Вероятность",
        xlim=(0.0, 1.4),
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    for field, color in ((30.0, "#1b9e77"), (40.0, "#d95f02")):
        x, profile = profiles[field]
        metric = next(
            row for row in penumbra_rows if row["nominal_field_mm"] == field
        )
        ax.plot(
            x,
            profile,
            color=color,
            linewidth=2.0,
            label=(
                f"{field:.0f}×{field:.0f} мм; "
                f"полутень {metric['penumbra_80_20_mm']:.1f} мм"
            ),
        )
    ax.axhline(0.8, color="#666666", linestyle=":", linewidth=1.0)
    ax.axhline(0.2, color="#666666", linestyle=":", linewidth=1.0)
    ax.set(
        title=(
            "Геометрический профиль в изоцентре\n"
            "опубликованный максимум полутени РОКУС-АМ: 17,5 мм"
        ),
        xlabel="Поперечная координата, мм",
        ylabel="Относительный флюенс",
        xlim=(-45.0, 45.0),
        ylim=(0.0, 1.08),
    )
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    comparison = rows[:4]
    labels = [
        "две линии\nпараллельно",
        "геометрия\nРОКУС",
        "спектр\nРОКУС",
        "полная модель\nРОКУС",
    ]
    let_values = [float(row["gtv_LETd_w_keV_um"]) for row in comparison]
    bars = ax.bar(
        np.arange(4),
        let_values,
        color=["#777777", "#7570b3", "#66c2a5", "#e6ab02"],
    )
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set(
        title="ЛПЭ: изменение связано со спектром, а не с расходимостью",
        ylabel=r"$LET_{d,w}$, кэВ/мкм",
        xticks=np.arange(4),
        xticklabels=labels,
        ylim=(0.30, 0.39),
    )
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    ax.plot(
        baseline_x,
        baseline_y,
        color="#777777",
        linewidth=2.0,
        label=(
            "параллельное поле, "
            f"D90={rows[0]['report_D90_over_Dmean']:.3f}"
        ),
    )
    ax.plot(
        rokus_x,
        rokus_y,
        color="#e6ab02",
        linewidth=2.2,
        label=(
            "РОКУС 30×30 мм, "
            f"D90={rows[3]['report_D90_over_Dmean']:.3f}"
        ),
    )
    ax.set(
        title=(
            "Pooled DVH на отчётной сетке\n"
            "число историй подобрано по статистике энерговклада"
        ),
        xlabel=r"$D/D_{mean}$",
        ylabel="Объём GTVp, получающий ≥ D, %",
        xlim=(0.0, 2.1),
        ylim=(0.0, 101.0),
    )
    ax.grid(alpha=0.25)
    ax.legend()

    fig.suptitle(
        "Co-60 РОКУС-АМ: литературно ограниченная модель источника",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            args.output_dir / f"gamma_rokus_literature_surrogate.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    result = {
        "status": "literature_bounded_surrogate_not_machine_commissioning",
        "source_spectrum_probability_sum": float(np.sum(probabilities)),
        "source_spectrum_mean_MeV": float(
            np.sum(energies * probabilities)
        ),
        "scenarios": rows,
        "geometric_penumbra": penumbra_rows,
        "limitations": [
            "no material source capsule or treatment head",
            "no jaw transmission, leakage or head scatter",
            "absolute dose per accepted photon is not dose per Co-60 decay",
            "no measured profiles or output factors for the local unit",
        ],
    }
    (
        args.output_dir / "gamma_rokus_literature_surrogate.json"
    ).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spectrum-csv", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-dvh", type=Path, required=True)
    parser.add_argument("--rokus-30-summary", type=Path, required=True)
    parser.add_argument("--rokus-30-dvh", type=Path, required=True)
    parser.add_argument("--rokus-40-summary", type=Path, required=True)
    parser.add_argument("--geometry-only-summary", type=Path, required=True)
    parser.add_argument("--spectrum-only-summary", type=Path, required=True)
    args = parser.parse_args()
    rows = analyse(args)
    print(json.dumps(rows, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
