"""Plot the representative C-12 through-field smoke test in the rat GTVp."""

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


DEFAULT_INPUT = Path(
    r"C:\dev\dissertation\task4_5"
    r"\scoring_v2_carbon_c12_rat_through_smoke\gtv_analysis"
)

COMPONENT_LABELS = {
    "electron": r"$e^-$",
    "hydrogen": "H",
    "helium": "He",
    "ion_Z3_to_Z6": r"Li–C ($Z=3$–6)",
    "ion_Z7_plus": r"$Z\geq7$",
    "primary_C12": r"первичный $^{12}$C",
}
COMPONENT_COLORS = {
    "electron": "#5c6bc0",
    "hydrogen": "#26a69a",
    "helium": "#f9a825",
    "ion_Z3_to_Z6": "#d84315",
    "ion_Z7_plus": "#6d4c41",
    "primary_C12": "#ad1457",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def float_value(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-stem", type=Path)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_stem = (
        args.output_stem.resolve()
        if args.output_stem
        else input_dir / "carbon_c12_rat_through_smoke"
    )
    summary = read_rows(input_dir / "gtv_summary.csv")[0]
    profile_rows = read_rows(
        input_dir / "through_central_depth_profile.csv"
    )
    component_rows = read_rows(
        input_dir / "gtv_component_breakdown.csv"
    )

    depth = np.asarray(
        [
            float_value(row, "depth_mm_from_plus_y_surface")
            for row in profile_rows
        ]
    )
    energy = np.asarray(
        [
            float_value(row, "central_energy_smoothed_keV")
            for row in profile_rows
        ]
    )
    normalised_energy = energy / np.nanmax(energy)
    gtv_low = float_value(
        summary, "gtv_depth_min_mm_from_plus_y_surface"
    )
    gtv_high = float_value(
        summary, "gtv_depth_max_mm_from_plus_y_surface"
    )
    peak_depth = float_value(summary, "central_peak_depth_mm")
    gtv_let = float_value(summary, "gtv_LETd_w_keV_um")
    coverage = 100.0 * float_value(
        summary, "gtv_nonzero_dose_fraction"
    )

    components = [
        row for row in component_rows
        if float_value(row, "fraction_of_total_gtv_depenergy") > 0.0
    ]
    components.sort(
        key=lambda row: float_value(
            row, "fraction_of_total_gtv_depenergy"
        )
    )
    component_names = [
        COMPONENT_LABELS.get(row["component"], row["component"])
        for row in components
    ]
    component_fraction = np.asarray(
        [
            100.0 * float_value(
                row, "fraction_of_total_gtv_depenergy"
            )
            for row in components
        ]
    )
    component_let = np.asarray(
        [
            float_value(row, "component_gtv_LETd_w_keV_um")
            for row in components
        ]
    )
    component_colors = [
        COMPONENT_COLORS.get(row["component"], "#777777")
        for row in components
    ]

    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        plt.rcParams.update(
            {
                "font.size": 10.5,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
            }
        )
        figure, (ax_depth, ax_components) = plt.subplots(
            1,
            2,
            figsize=(12.0, 4.9),
            constrained_layout=True,
            gridspec_kw={"width_ratios": (1.55, 1.0)},
        )

        ax_depth.axvspan(
            gtv_low,
            gtv_high,
            color="#e53935",
            alpha=0.13,
            label=f"GTVp: {gtv_low:.1f}–{gtv_high:.1f} мм",
        )
        ax_depth.plot(
            depth,
            normalised_energy,
            color="#263238",
            linewidth=2.1,
            label="сглаженный энерговклад",
        )
        ax_depth.axvline(
            peak_depth,
            color="#c62828",
            linestyle="--",
            linewidth=1.2,
            label=f"локальный максимум {peak_depth:.1f} мм",
        )
        ax_depth.annotate(
            "направление пучка",
            xy=(13.5, 1.02),
            xytext=(2.0, 1.02),
            arrowprops={"arrowstyle": "->", "color": "0.35"},
            color="0.35",
            va="center",
            fontsize=9,
        )
        ax_depth.set(
            xlabel=(
                "Глубина от входной грани фантома, мм\n"
                "(пучок: Geant4 +y → −y)"
            ),
            ylabel="Энерговклад / максимум",
            title="Центральный профиль в КТ-фантоме крысы",
            xlim=(0.0, 51.2),
            ylim=(0.0, 1.08),
        )
        ax_depth.grid(alpha=0.2)
        ax_depth.legend(loc="lower left", frameon=True, fontsize=9)

        y = np.arange(len(components))
        bars = ax_components.barh(
            y,
            component_fraction,
            color=component_colors,
            alpha=0.9,
        )
        ax_components.set_yticks(y, component_names)
        ax_components.set_xlim(0.0, 112.0)
        ax_components.set(
            xlabel="Доля энерговклада в GTVp, %",
            title=(
                "Компоненты в GTVp\n"
                rf"полная $LET_D={gtv_let:.2f}$ кэВ/мкм"
            ),
        )
        ax_components.grid(axis="x", alpha=0.2)
        for bar, fraction, let_value in zip(
            bars, component_fraction, component_let
        ):
            ax_components.text(
                min(fraction + 1.2, 105.0),
                bar.get_y() + bar.get_height() / 2.0,
                (
                    f"{fraction:.2f}%"
                    + (
                        rf"; $LET_D={let_value:.2f}$"
                        if np.isfinite(let_value)
                        else ""
                    )
                ),
                va="center",
                ha="left",
                fontsize=9,
            )

        figure.suptitle(
            (
                r"Пилот $^{12}$C, 434 МэВ/нуклон: "
                "репрезентативная GTVp, без геометрии стенда\n"
                f"200 первичных ионов; ненулевая доза в "
                f"{coverage:.1f}% тонких вокселей — DVH не оценивается"
            ),
            fontsize=13,
        )
        for suffix in (".png", ".svg", ".pdf"):
            figure.savefig(output_stem.with_suffix(suffix), dpi=260)
        plt.close(figure)
    finally:
        configurator.restore_original_styles()

    print(f"output={output_stem}")
    print(f"gtv_LETd_w_keV_um={gtv_let}")
    print(f"nonzero_voxel_fraction={coverage / 100.0}")


if __name__ == "__main__":
    main()
