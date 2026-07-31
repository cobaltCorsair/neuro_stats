"""Compare Opt3 and Opt4 for the archived eight-energy C-12 source."""

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


DEFAULT_ROOT = Path(
    r"C:\dev\dissertation\task4_5"
    r"\scoring_v2_carbon_c12_rat_sobp_8energy_smoke"
)
CASES = ("opt3_1000", "opt4_1000")
CASE_LABELS = {
    "opt3_1000": "Opt3",
    "opt4_1000": "Opt4",
}
CASE_COLORS = {
    "opt3_1000": "#00796b",
    "opt4_1000": "#ef6c00",
}
TOTAL_ENERGIES_MEV = np.asarray(
    [618.0, 657.3, 696.6, 735.9, 775.2, 811.9, 844.3, 876.7]
)
ENERGIES_MEV_U = TOTAL_ENERGIES_MEV / 12.0
WEIGHTS = np.asarray([0.20, 0.37, 0.45, 0.54, 0.66, 0.85, 1.00, 2.00])


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-stem", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    output_stem = (
        args.output_stem.resolve()
        if args.output_stem
        else root / "carbon_c12_archived_source_opt3_opt4_1000"
    )
    summaries: dict[str, dict[str, str]] = {}
    profiles: dict[str, list[dict[str, str]]] = {}
    components: dict[str, list[dict[str, str]]] = {}
    for case in CASES:
        analysis = root / case / "gtv_analysis"
        summaries[case] = read_rows(analysis / "gtv_summary.csv")[0]
        profiles[case] = read_rows(
            analysis / "through_central_depth_profile.csv"
        )
        components[case] = read_rows(
            analysis / "gtv_component_breakdown.csv"
        )

    gtv_low = number(
        summaries[CASES[0]], "gtv_depth_min_mm_from_plus_y_surface"
    )
    gtv_high = number(
        summaries[CASES[0]], "gtv_depth_max_mm_from_plus_y_surface"
    )

    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        plt.rcParams.update(
            {
                "font.size": 10.5,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "legend.fontsize": 9,
            }
        )
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(12.2, 8.4),
            constrained_layout=True,
        )
        ax_source, ax_depth, ax_let, ax_components = axes.ravel()

        normalised_weights = WEIGHTS / np.sum(WEIGHTS)
        bars = ax_source.bar(
            ENERGIES_MEV_U,
            100.0 * normalised_weights,
            width=2.25,
            color="#455a64",
            alpha=0.88,
        )
        ax_source.set(
            xlabel="Энергия, МэВ/нуклон",
            ylabel="Нормированная вероятность, %",
            title=(
                "Архивный источник: 8 моноэнергий\n"
                "(GPS получает 618–876,7 МэВ на ядро)"
            ),
            xlim=(49.0, 76.0),
        )
        ax_source.set_ylim(
            0.0,
            float(np.max(100.0 * normalised_weights)) * 1.22,
        )
        ax_source.grid(axis="y", alpha=0.2)
        for bar, energy in zip(bars, ENERGIES_MEV_U):
            ax_source.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.45,
                f"{energy:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

        ax_depth.axvspan(
            gtv_low,
            gtv_high,
            color="#e53935",
            alpha=0.11,
            label=f"GTVp: {gtv_low:.1f}–{gtv_high:.1f} мм",
        )
        for case in CASES:
            rows = profiles[case]
            depth = np.asarray(
                [
                    number(row, "depth_mm_from_plus_y_surface")
                    for row in rows
                ]
            )
            energy = np.asarray(
                [
                    number(row, "central_energy_smoothed_keV")
                    for row in rows
                ]
            )
            ax_depth.plot(
                depth,
                energy / np.nanmax(energy),
                color=CASE_COLORS[case],
                linewidth=2.0,
                label=CASE_LABELS[case],
            )
            peak_depth = number(
                summaries[case], "central_peak_depth_mm"
            )
            ax_depth.axvline(
                peak_depth,
                color=CASE_COLORS[case],
                linestyle=":",
                linewidth=1.1,
            )
        ax_depth.set(
            xlabel="Глубина от входной грани КТ-фантома, мм",
            ylabel="Энерговклад / максимум",
            title="Центральный профиль в фантоме крысы",
            xlim=(0.0, 51.2),
            ylim=(0.0, 1.08),
        )
        ax_depth.grid(alpha=0.2)
        ax_depth.legend(loc="lower left", frameon=True)

        x = np.arange(len(CASES))
        total_let = np.asarray(
            [
                number(summaries[case], "gtv_LETd_w_keV_um")
                for case in CASES
            ]
        )
        let_bars = ax_let.bar(
            x,
            total_let,
            width=0.56,
            color=[CASE_COLORS[case] for case in CASES],
            alpha=0.9,
        )
        ax_let.set_xticks(x, [CASE_LABELS[case] for case in CASES])
        ax_let.set(
            ylabel=r"$LET_D$ GTVp, кэВ/мкм",
            title="Интегральная ЛПЭ в GTVp",
            ylim=(0.0, float(np.max(total_let)) * 1.18),
        )
        ax_let.grid(axis="y", alpha=0.2)
        ax_let.bar_label(
            let_bars,
            labels=[f"{value:.1f}" for value in total_let],
            padding=4,
            fontsize=10,
        )
        relative_difference = (
            100.0 * (total_let[1] - total_let[0]) / total_let[0]
        )
        ax_let.text(
            0.5,
            float(np.max(total_let)) * 1.105,
            f"Opt4 − Opt3: {relative_difference:+.2f}%",
            ha="center",
            va="center",
            fontsize=9.5,
        )

        primary_fractions = []
        secondary_fractions = []
        for case in CASES:
            by_component = {
                row["component"]: number(
                    row, "fraction_of_total_gtv_depenergy"
                )
                for row in components[case]
            }
            primary = 100.0 * by_component.get("primary_C12", 0.0)
            primary_fractions.append(primary)
            secondary_fractions.append(100.0 - primary)
        primary_fractions = np.asarray(primary_fractions)
        secondary_fractions = np.asarray(secondary_fractions)
        primary_bars = ax_components.bar(
            x,
            primary_fractions,
            width=0.56,
            color="#ad1457",
            label=r"первичный $^{12}$C",
        )
        secondary_bars = ax_components.bar(
            x,
            secondary_fractions,
            width=0.56,
            bottom=primary_fractions,
            color="#90a4ae",
            label="все вторичные компоненты",
        )
        ax_components.set_xticks(
            x, [CASE_LABELS[case] for case in CASES]
        )
        ax_components.set(
            ylabel="Доля энерговклада GTVp, %",
            title="Первичный трек отделён от вторичных частиц",
            ylim=(0.0, 106.0),
        )
        ax_components.grid(axis="y", alpha=0.2)
        for index, (primary, secondary) in enumerate(
            zip(primary_fractions, secondary_fractions)
        ):
            ax_components.text(
                index,
                primary / 2.0,
                f"первичный $^{{12}}$C\n{primary:.2f}%",
                color="white",
                ha="center",
                va="center",
                fontsize=9.5,
            )
            ax_components.text(
                index,
                primary + secondary / 2.0,
                f"вторичные\n{secondary:.2f}%",
                color="#263238",
                ha="center",
                va="center",
                fontsize=8.5,
            )

        figure.suptitle(
            (
                r"$^{12}$C в КТ-фантоме крысы: "
                "архивный восьмиэнергетический источник\n"
                "1000 первичных ионов на вариант; "
                "QGSP_INCLXX; DVH не интерпретируется"
            ),
            fontsize=14,
        )
        for suffix in (".png", ".svg", ".pdf"):
            figure.savefig(output_stem.with_suffix(suffix), dpi=260)
        plt.close(figure)
    finally:
        configurator.restore_original_styles()

    print(f"output={output_stem}")
    print(f"LETd_Opt3={total_let[0]}")
    print(f"LETd_Opt4={total_let[1]}")
    print(f"relative_difference_percent={relative_difference}")


if __name__ == "__main__":
    main()
