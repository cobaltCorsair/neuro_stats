"""Compare makerP with the existing 100-MeV monoenergetic scenarios."""

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
DEFAULT_MONO_DIR = BASE_OUTPUT / "proton_100MeV_reduced_50k"
DEFAULT_MAKERP_DIR = BASE_OUTPUT / "proton_makerP_vacuum_200k"
DEFAULT_OUTPUT_DIR = BASE_OUTPUT / "proton_makerP_comparison"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_profile(path: Path) -> dict[str, np.ndarray]:
    rows = read_rows(path)
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in rows[0]
    }


def maker_case(maker_dir: Path) -> str:
    rows = read_rows(maker_dir / "gtv_summary.csv")
    if len(rows) != 1:
        raise ValueError(
            f"Expected one makerP row in {maker_dir}, found {len(rows)}"
        )
    return rows[0]["case"]


def load_cases(
    mono_dir: Path,
    maker_dir: Path,
) -> tuple[list[dict[str, str]], dict[str, dict[str, np.ndarray]]]:
    mono_rows = read_rows(mono_dir / "gtv_summary.csv")
    selected_mono = [
        row for row in mono_rows if row["case"] in {"through", "peak45", "peak55"}
    ]
    maker_rows = read_rows(maker_dir / "gtv_summary.csv")
    if len(maker_rows) != 1:
        raise ValueError("makerP summary must contain exactly one row")
    rows = selected_mono + maker_rows

    profiles: dict[str, dict[str, np.ndarray]] = {}
    for row in selected_mono:
        case = row["case"]
        profiles[case] = read_profile(
            mono_dir / f"{case}_central_depth_profile.csv"
        )
    maker_name = maker_rows[0]["case"]
    profiles[maker_name] = read_profile(
        maker_dir / f"{maker_name}_central_depth_profile.csv"
    )
    return rows, profiles


def plot_comparison(
    rows: list[dict[str, str]],
    profiles: dict[str, dict[str, np.ndarray]],
    output_stem: Path,
) -> None:
    cases = [row["case"] for row in rows]
    labels = {
        "through": "прострел\n100 МэВ",
        "peak45": "пик 45 мм\n100 МэВ",
        "peak55": "пик 55 мм\n100 МэВ",
    }
    for case in cases:
        if case.startswith("makerP"):
            labels[case] = "makerP\n16–51 МэВ"

    colours = {
        "through": "#4477aa",
        "peak45": "#66c2a5",
        "peak55": "#ee8866",
    }
    for case in cases:
        if case.startswith("makerP"):
            colours[case] = "#6a3d9a"

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
            figsize=(12.6, 9.2),
            constrained_layout=True,
        )
        ax_depth, ax_coverage, ax_let, ax_energy = axes.ravel()

        gtv_low = float(rows[0]["gtv_depth_min_mm_from_plus_y_surface"])
        gtv_high = float(rows[0]["gtv_depth_max_mm_from_plus_y_surface"])
        ax_depth.axvspan(
            gtv_low,
            gtv_high,
            color="#e31a1c",
            alpha=0.10,
            label="аксиальный интервал GTVp",
        )
        for row in rows:
            case = row["case"]
            profile = profiles[case]
            smoothed = profile["central_energy_smoothed_keV"]
            maximum = float(np.max(smoothed))
            ax_depth.plot(
                profile["depth_mm_from_plus_y_surface"],
                smoothed / maximum,
                color=colours[case],
                linewidth=2.2,
                label=labels[case].replace("\n", " "),
            )
        ax_depth.set(
            xlabel="Глубина от поверхности лапы, мм",
            ylabel="Нормированный энерговклад",
            title="Центральный глубинный профиль",
            xlim=(0.0, 51.2),
            ylim=(0.0, 1.08),
        )
        ax_depth.grid(alpha=0.22)
        ax_depth.legend(frameon=True, ncol=2)

        x = np.arange(len(rows))
        d90 = np.asarray(
            [
                float(
                    row[
                        "gtv_aggregated_D90_over_Dmean_1p6x1p6x0p8"
                    ]
                )
                for row in rows
            ]
        )
        d50 = np.asarray(
            [
                float(
                    row[
                        "gtv_aggregated_D50_over_Dmean_1p6x1p6x0p8"
                    ]
                )
                for row in rows
            ]
        )
        width = 0.34
        d90_bars = ax_coverage.bar(
            x - width / 2,
            d90,
            width,
            color="#67a9cf",
            label=r"$D_{90}/D_{\mathrm{mean}}$",
        )
        d50_bars = ax_coverage.bar(
            x + width / 2,
            d50,
            width,
            color="#ef8a62",
            label=r"$D_{50}/D_{\mathrm{mean}}$",
        )
        ax_coverage.set_xticks(x, [labels[case] for case in cases])
        ax_coverage.set(
            ylabel="Относительная доза",
            title="Покрытие GTVp после усреднения 1,6×1,6×0,8 мм",
            ylim=(0.0, 1.15),
            xlim=(-0.65, len(rows) - 0.35),
        )
        ax_coverage.grid(axis="y", alpha=0.22)
        ax_coverage.legend(frameon=True, loc="lower right")
        ax_coverage.bar_label(d90_bars, fmt="%.2f", padding=3, fontsize=9)
        ax_coverage.bar_label(d50_bars, fmt="%.2f", padding=3, fontsize=9)

        let_values = np.asarray(
            [float(row["gtv_LETd_w_keV_um"]) for row in rows]
        )
        let_bars = ax_let.bar(
            x,
            let_values,
            color=[colours[case] for case in cases],
            alpha=0.88,
        )
        ax_let.set_xticks(x, [labels[case] for case in cases])
        ax_let.set(
            ylabel="Дозо-взвешенная ЛПЭ, кэВ/мкм",
            title="Полная ЛПЭ внутри GTVp",
            ylim=(0.0, max(let_values) * 1.20),
            xlim=(-0.65, len(rows) - 0.35),
        )
        ax_let.grid(axis="y", alpha=0.22)
        ax_let.bar_label(let_bars, fmt="%.2f", padding=3, fontsize=9)

        energy_fraction = 100.0 * np.asarray(
            [float(row["gtv_energy_fraction_of_phantom"]) for row in rows]
        )
        energy_bars = ax_energy.bar(
            x,
            energy_fraction,
            color=[colours[case] for case in cases],
            alpha=0.88,
        )
        ax_energy.set_xticks(x, [labels[case] for case in cases])
        ax_energy.set(
            ylabel="Доля энерговклада фантома в GTVp, %",
            title="Пространственная концентрация энерговклада",
            ylim=(0.0, max(energy_fraction) * 1.22),
            xlim=(-0.65, len(rows) - 0.35),
        )
        ax_energy.grid(axis="y", alpha=0.22)
        ax_energy.bar_label(energy_bars, fmt="%.1f%%", padding=3, fontsize=9)

        fig.suptitle(
            "Многокомпонентный источник makerP и моноэнергетические "
            "протонные сценарии",
            fontsize=15,
        )
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(output_stem.with_suffix(suffix), dpi=300)
        plt.close(fig)
    finally:
        configurator.restore_original_styles()


def write_comparison_csv(
    rows: list[dict[str, str]],
    path: Path,
) -> None:
    fields = (
        "case",
        "configuration",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mono-dir", type=Path, default=DEFAULT_MONO_DIR)
    parser.add_argument("--makerp-dir", type=Path, default=DEFAULT_MAKERP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows, profiles = load_cases(args.mono_dir, args.makerp_dir)
    plot_comparison(
        rows,
        profiles,
        args.output_dir / "proton_makerP_vs_mono_comparison",
    )
    write_comparison_csv(
        rows,
        args.output_dir / "proton_makerP_vs_mono_comparison.csv",
    )
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
