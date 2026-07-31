"""Plot the validated scientific-supervisor makerP proton source."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


NEURO_STATS_ROOT = Path(__file__).resolve().parents[3]
if str(NEURO_STATS_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURO_STATS_ROOT))

from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (  # noqa: E402
    MatplotlibConfigurator,
)


DEFAULT_SOURCE_DIR = Path(
    r"C:\dev\dissertation\task4_5\source_models\makerP"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\dev\dissertation\task4_5\outputs"
    r"\geant4_livermore_20260724\proton_makerP_source"
)


def read_numeric_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def plot_source(
    source_rows: list[dict[str, float]],
    energy_rows: list[dict[str, float]],
    output_stem: Path,
) -> None:
    energy = np.asarray([row["energy_MeV"] for row in energy_rows])
    probability = np.asarray([row["probability"] for row in energy_rows])
    source_energy = np.asarray(
        [row["energy_MeV"] for row in source_rows]
    )
    source_probability = np.asarray(
        [row["probability"] for row in source_rows]
    )
    source_x = np.asarray(
        [row["rat_isocentre_x_mm"] for row in source_rows]
    )
    source_z = np.asarray(
        [row["rat_isocentre_z_mm"] for row in source_rows]
    )
    source_radius = np.hypot(source_x, source_z)
    source_sigma_x = np.asarray(
        [row["sigma_x_mm"] for row in source_rows]
    )
    source_sigma_y = np.asarray(
        [row["sigma_y_mm"] for row in source_rows]
    )

    sigma_x = np.asarray(
        [
            np.average(
                source_sigma_x[source_energy == value],
                weights=source_probability[source_energy == value],
            )
            for value in energy
        ]
    )
    sigma_y = np.asarray(
        [
            np.average(
                source_sigma_y[source_energy == value],
                weights=source_probability[source_energy == value],
            )
            for value in energy
        ]
    )
    radius_min = np.asarray(
        [
            np.min(source_radius[source_energy == value])
            for value in energy
        ]
    )
    radius_max = np.asarray(
        [
            np.max(source_radius[source_energy == value])
            for value in energy
        ]
    )

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
            figsize=(12.4, 9.2),
            constrained_layout=True,
        )
        ax_spectrum, ax_field, ax_sigma, ax_radius = axes.ravel()

        widths = np.diff(energy, prepend=energy[0] - 1.5)
        widths = np.minimum(widths * 0.76, 1.45)
        bars = ax_spectrum.bar(
            energy,
            100.0 * probability,
            width=widths,
            color="#4477aa",
            alpha=0.88,
        )
        max_index = int(np.argmax(probability))
        ax_spectrum.annotate(
            f"{100.0 * probability[max_index]:.1f}%",
            xy=(energy[max_index], 100.0 * probability[max_index]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
        ax_spectrum.axvline(
            np.average(energy, weights=probability),
            color="#aa3377",
            linestyle="--",
            linewidth=1.6,
            label="средневзвешенная: 41,07 МэВ",
        )
        ax_spectrum.set(
            xlabel="Энергия компоненты, МэВ",
            ylabel="Доля первичных частиц, %",
            title="Дискретный энергетический спектр",
            xlim=(14.5, 52.5),
        )
        ax_spectrum.set_ylim(
            0.0,
            max(bar.get_height() for bar in bars) * 1.22,
        )
        ax_spectrum.grid(axis="y", alpha=0.22)
        ax_spectrum.legend(frameon=True, loc="upper left")

        marker_size = 28.0 + 330.0 * np.sqrt(
            source_probability / np.max(source_probability)
        )
        scatter = ax_field.scatter(
            source_x,
            source_z,
            c=source_energy,
            s=marker_size,
            cmap="viridis",
            alpha=0.82,
            edgecolors="0.2",
            linewidths=0.35,
        )
        ax_field.add_patch(
            Circle(
                (0.0, 0.0),
                14.4,
                fill=False,
                color="#cc3311",
                linestyle="--",
                linewidth=1.4,
                label="радиус прежнего поля 14,4 мм",
            )
        )
        ax_field.axhline(0.0, color="0.75", linewidth=0.8)
        ax_field.axvline(0.0, color="0.75", linewidth=0.8)
        ax_field.set(
            xlabel="x в плоскости изоцентра, мм",
            ylabel="z в плоскости изоцентра, мм",
            title="Центральные лучи 199 компонент",
            xlim=(-29.0, 29.0),
            ylim=(-29.0, 29.0),
            aspect="equal",
        )
        ax_field.grid(alpha=0.16)
        ax_field.legend(frameon=True, loc="lower left")
        colorbar = fig.colorbar(scatter, ax=ax_field, pad=0.02)
        colorbar.set_label("Энергия, МэВ")

        ax_sigma.plot(
            energy,
            sigma_x,
            marker="o",
            linewidth=1.9,
            label=r"$\sigma_x$",
        )
        ax_sigma.plot(
            energy,
            sigma_y,
            marker="s",
            linewidth=1.9,
            label=r"$\sigma_y$",
        )
        ax_sigma.set(
            xlabel="Энергия компоненты, МэВ",
            ylabel="Стандартное отклонение, мм",
            title="Поперечный размер пучка",
            xlim=(14.5, 52.5),
        )
        ax_sigma.grid(alpha=0.22)
        ax_sigma.legend(frameon=True)

        ax_radius.fill_between(
            energy,
            radius_min,
            radius_max,
            color="#66c2a5",
            alpha=0.32,
            label="диапазон компонент",
        )
        ax_radius.plot(
            energy,
            radius_max,
            color="#238b45",
            marker="o",
            linewidth=1.8,
            label="максимальный радиус",
        )
        ax_radius.axhline(
            14.4,
            color="#cc3311",
            linestyle="--",
            linewidth=1.4,
            label="14,4 мм",
        )
        ax_radius.set(
            xlabel="Энергия компоненты, МэВ",
            ylabel="Радиус центрального луча, мм",
            title="Геометрия поля в изоцентре",
            xlim=(14.5, 52.5),
            ylim=(0.0, 27.0),
        )
        ax_radius.grid(alpha=0.22)
        ax_radius.legend(frameon=True, loc="lower left")

        fig.suptitle(
            "Авторский GPS-источник makerP после жёсткого поворота "
            "к геометрии крысы",
            fontsize=15,
        )
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(output_stem.with_suffix(suffix), dpi=300)
        plt.close(fig)
    finally:
        configurator.restore_original_styles()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=DEFAULT_SOURCE_DIR / "makerP_sources.csv",
    )
    parser.add_argument(
        "--energy-csv",
        type=Path,
        default=DEFAULT_SOURCE_DIR / "makerP_energy_spectrum.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()

    source_rows = read_numeric_rows(args.source_csv)
    energy_rows = read_numeric_rows(args.energy_csv)
    if len(source_rows) != 199:
        raise ValueError(f"Expected 199 sources, found {len(source_rows)}")
    if len(energy_rows) != 19:
        raise ValueError(
            f"Expected 19 energy groups, found {len(energy_rows)}"
        )
    plot_source(
        source_rows,
        energy_rows,
        args.output_dir / "proton_makerP_source_validation",
    )
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
