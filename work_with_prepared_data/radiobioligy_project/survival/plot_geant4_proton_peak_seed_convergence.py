"""Plot between-seed stability of the Geant4 60 MeV proton pilot."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (
    MatplotlibConfigurator,
)


GTV_MIN_MM = 16.85
GTV_MAX_MM = 33.45


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        field: np.asarray(
            [
                float(row[field]) if row[field].strip() else np.nan
                for row in rows
            ],
            dtype=float,
        )
        for field in rows[0]
    }


def read_metrics(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: list[dict[str, float | str]] = []
    for row in rows:
        result.append(
            {
                key: value if key in {"seed_label", "run_directory"}
                else float(value)
                for key, value in row.items()
            }
        )
    return result


def cv_percent(
    metrics: list[dict[str, float | str]],
    field: str,
) -> float:
    values = [float(row[field]) for row in metrics]
    return 100.0 * statistics.stdev(values) / abs(statistics.fmean(values))


def plot(
    runs: list[tuple[str, Path]],
    metrics: list[dict[str, float | str]],
    output_stem: Path,
) -> None:
    profiles = [
        (
            label,
            read_numeric_csv(
                directory / "central_axis_r2mm_depth_profile.csv"
            ),
        )
        for label, directory in runs
    ]
    colors = ["#3366a8", "#d17828", "#3f8f5f", "#8b5aa5"]
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        fig = plt.figure(figsize=(15, 13), constrained_layout=True)
        grid = fig.add_gridspec(
            2,
            2,
            height_ratios=(1.0, 1.15),
            width_ratios=(1.25, 0.75),
        )
        ax_profile = fig.add_subplot(grid[0, 0])
        ax_roi = fig.add_subplot(grid[0, 1])
        ax_cv = fig.add_subplot(grid[1, :])

        ax_profile.axvspan(
            GTV_MIN_MM,
            GTV_MAX_MM,
            color="#dedede",
            alpha=0.55,
            label="GTVp",
            zorder=0,
        )
        for (label, profile), color in zip(profiles, colors):
            depth = profile["depth_mm"]
            energy = profile["smoothed_depEnergy_keV"]
            in_view = (depth >= 15.5) & (depth <= 36.5)
            ax_profile.plot(
                depth[in_view],
                energy[in_view] / np.nanmax(energy[in_view]),
                linewidth=2.2,
                color=color,
                label=label,
            )
        ax_profile.set(
            xlabel="Глубина от входной границы фантома, мм",
            ylabel="Нормированный энерговклад",
            ylim=(0, 1.08),
            title="Центральный профиль (r = 2 мм)",
        )
        ax_profile.grid(alpha=0.25)
        ax_profile.legend(loc="lower left", frameon=True, ncol=2)

        x = np.arange(len(metrics), dtype=float)
        all_let = np.asarray(
            [float(row["gtv_LETd_all_keV_um"]) for row in metrics]
        )
        hydrogen_let = np.asarray(
            [float(row["gtv_LETd_hydrogen_keV_um"]) for row in metrics]
        )
        ax_roi.scatter(
            x - 0.08,
            all_let,
            s=85,
            color="#555555",
            marker="o",
            label="все заряженные",
            zorder=3,
        )
        ax_roi.scatter(
            x + 0.08,
            hydrogen_let,
            s=85,
            color="#2171b5",
            marker="s",
            label="водород",
            zorder=3,
        )
        ax_roi.axhline(
            np.mean(all_let),
            color="#555555",
            linestyle=":",
            linewidth=1.5,
        )
        ax_roi.axhline(
            np.mean(hydrogen_let),
            color="#2171b5",
            linestyle=":",
            linewidth=1.5,
        )
        ax_roi.set_xticks(
            x, [str(row["seed_label"]) for row in metrics]
        )
        ax_roi.set(
            ylabel=r"$LET_D$ во всей GTVp, кэВ/мкм",
            title="Интегральная ROI-метрика",
            ylim=(0, max(6.2, 1.15 * np.max(all_let))),
        )
        ax_roi.grid(axis="y", alpha=0.25)
        ax_roi.legend(loc="upper left", frameon=True)

        cv_fields = [
            ("gtv_depenergy_weighted_depth_mm", "Средняя глубина в GTVp"),
            ("gtv_LETd_hydrogen_keV_um", r"$LET_D$(H) во всей GTVp"),
            ("gtv_mean_dose_Gy_per_run", "Средняя доза в GTVp"),
            ("gtv_coverage_fraction", "Покрытие GTVp"),
            ("energy_peak_depth_mm", "Глубина локального пика"),
            (
                "central_mean_dose_at_energy_peak_Gy_per_run",
                "Доза в локальном пике",
            ),
            ("gtv_LETd_all_keV_um", r"$LET_D$(все) во всей GTVp"),
            (
                "central_peak_LETd_hydrogen_keV_um",
                r"$LET_D$(H) в локальном пике",
            ),
            (
                "central_peak_LETd_all_keV_um",
                r"$LET_D$(все) в локальном пике",
            ),
        ]
        cv_values = [cv_percent(metrics, field) for field, _ in cv_fields]
        labels = [label for _, label in cv_fields]
        bar_colors = [
            "#4c9a68" if value < 10.0
            else "#d6a441" if value < 25.0
            else "#c75b52"
            for value in cv_values
        ]
        y = np.arange(len(labels))
        bars = ax_cv.barh(y, cv_values, color=bar_colors, alpha=0.9)
        ax_cv.axvline(
            10.0,
            color="#333333",
            linestyle="--",
            linewidth=1.3,
            label="ориентир CV = 10%",
        )
        ax_cv.set_yticks(y, labels)
        ax_cv.tick_params(axis="y", labelsize=16)
        ax_cv.invert_yaxis()
        ax_cv.set(
            xlabel="CV между тремя seed, %",
            title="Устойчивость выходных показателей",
            xlim=(0, max(105.0, 1.10 * max(cv_values))),
        )
        ax_cv.grid(axis="x", alpha=0.25)
        ax_cv.legend(loc="upper right", frameon=True)
        ax_cv.bar_label(
            bars,
            labels=[f"{value:.1f}%" for value in cv_values],
            padding=4,
            fontsize=10,
        )

        fig.suptitle(
            "Протоны 60 МэВ, QGSP_INCLXX + G4EmLivermore: "
            "проверка трёх независимых seed\n"
            "1000 первичных протонов на seed; пилотный расчёт",
            fontsize=17,
        )
        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(
                output_stem.with_suffix(suffix),
                dpi=300 if suffix == ".png" else None,
                bbox_inches="tight",
            )
        plt.close(fig)
    finally:
        configurator.restore_original_styles()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot(
        args.run,
        read_metrics(args.metrics_csv),
        args.output_dir / "proton_peak_seed_convergence",
    )


if __name__ == "__main__":
    main()
