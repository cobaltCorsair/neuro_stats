"""Assess Monte-Carlo convergence of the makerP rat-phantom calculation."""

from __future__ import annotations

import argparse
import csv
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


BASE_OUTPUT = Path(
    r"C:\dev\dissertation\task4_5\outputs\geant4_livermore_20260724"
)
DEFAULT_INPUTS = (
    BASE_OUTPUT / "proton_makerP_vacuum_10k" / "gtv_summary.csv",
    BASE_OUTPUT / "proton_makerP_vacuum_50k" / "gtv_summary.csv",
    BASE_OUTPUT / "proton_makerP_vacuum_100k" / "gtv_summary.csv",
    BASE_OUTPUT / "proton_makerP_vacuum_200k" / "gtv_summary.csv",
)
DEFAULT_OUTPUT = BASE_OUTPUT / "proton_makerP_convergence"

FIELDS = (
    "histories",
    "central_peak_depth_mm",
    "gtv_nonzero_dose_fraction",
    "gtv_depenergy_MeV_per_primary",
    "gtv_energy_fraction_of_phantom",
    "gtv_D50_over_Dmean",
    "gtv_D90_over_Dmean",
    "gtv_aggregated_D50_over_Dmean_1p6x1p6x0p8",
    "gtv_aggregated_D90_over_Dmean_1p6x1p6x0p8",
    "gtv_dose_cv",
    "gtv_LETd_w_keV_um",
)


def read_summary(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(rows)}")
    return {field: float(rows[0][field]) for field in FIELDS}


def add_point_labels(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    fmt: str,
) -> None:
    for x_value, y_value in zip(x, y):
        axis.annotate(
            format(y_value, fmt),
            xy=(x_value, y_value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_convergence(
    rows: list[dict[str, float]],
    output_stem: Path,
) -> None:
    histories = np.asarray([row["histories"] for row in rows])
    coverage = np.asarray(
        [100.0 * row["gtv_nonzero_dose_fraction"] for row in rows]
    )
    d90_raw = np.asarray([row["gtv_D90_over_Dmean"] for row in rows])
    d90_aggregated = np.asarray(
        [
            row["gtv_aggregated_D90_over_Dmean_1p6x1p6x0p8"]
            for row in rows
        ]
    )
    energy = np.asarray(
        [row["gtv_depenergy_MeV_per_primary"] for row in rows]
    )
    let = np.asarray([row["gtv_LETd_w_keV_um"] for row in rows])

    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 12,
                "legend.fontsize": 9.5,
            }
        )
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12.2, 8.8),
            constrained_layout=True,
        )
        ax_coverage, ax_d90, ax_energy, ax_let = axes.ravel()
        common = {
            "marker": "o",
            "linewidth": 2.0,
            "markersize": 6.5,
        }

        ax_coverage.plot(histories, coverage, color="#4477aa", **common)
        add_point_labels(
            ax_coverage,
            histories,
            coverage,
            fmt=".1f",
        )
        ax_coverage.set(
            ylabel="Воксели GTVp с ненулевой дозой, %",
            title="Заполнение исходной сетки",
            ylim=(0.0, 108.0),
        )

        ax_d90.plot(
            histories,
            d90_raw,
            color="#ee8866",
            label="исходная сетка 0,4×0,4×0,2 мм",
            **common,
        )
        ax_d90.plot(
            histories,
            d90_aggregated,
            color="#228833",
            label="усреднение 1,6×1,6×0,8 мм",
            **common,
        )
        add_point_labels(ax_d90, histories, d90_raw, fmt=".2f")
        add_point_labels(
            ax_d90,
            histories,
            d90_aggregated,
            fmt=".2f",
        )
        ax_d90.set(
            ylabel=r"$D_{90}/D_{\mathrm{mean}}$",
            title="Сходимость объёмного покрытия",
            ylim=(0.0, 1.02),
        )
        ax_d90.legend(frameon=True, loc="lower right")

        ax_energy.plot(histories, energy, color="#66c2a5", **common)
        add_point_labels(ax_energy, histories, energy, fmt=".3f")
        spread = max(0.03, float(np.ptp(energy)) * 2.0)
        centre = float(np.mean(energy))
        ax_energy.set(
            ylabel="Энерговклад GTVp, МэВ/первичный",
            title="Интегральный энерговклад",
            ylim=(centre - spread, centre + spread),
        )

        ax_let.plot(histories, let, color="#6a3d9a", **common)
        add_point_labels(ax_let, histories, let, fmt=".2f")
        ax_let.set(
            ylabel="Дозо-взвешенная ЛПЭ, кэВ/мкм",
            title="ЛПЭ внутри GTVp",
            ylim=(min(let) * 0.92, max(let) * 1.08),
        )

        for axis in axes.ravel():
            axis.set_xscale("log")
            axis.set_xlabel("Число первичных протонов")
            axis.grid(alpha=0.22)
            axis.set_xticks(histories)
            axis.set_xticklabels(
                [f"{int(value / 1000)} тыс." for value in histories]
            )

        fig.suptitle(
            "Сходимость расчёта многокомпонентного источника makerP",
            fontsize=15,
        )
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(output_stem.with_suffix(suffix), dpi=300)
        plt.close(fig)
    finally:
        configurator.restore_original_styles()


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=DEFAULT_INPUTS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    missing = [path for path in args.inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing convergence inputs: "
            + ", ".join(str(path) for path in missing)
        )
    rows = sorted(
        [read_summary(path) for path in args.inputs],
        key=lambda row: row["histories"],
    )
    write_csv(
        rows,
        args.output_dir / "proton_makerP_convergence.csv",
    )
    plot_convergence(
        rows,
        args.output_dir / "proton_makerP_convergence",
    )
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
