"""Plot the primary and group-resolved neutron spectra for the rat GTV."""

from __future__ import annotations

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


TASK_DIR = Path(r"C:\dev\dissertation\task4_5")
INPUT = (
    TASK_DIR
    / "outputs"
    / "geant4_livermore_20260724"
    / "neutron_ng14_rat_kerma"
    / "neutron_gtv_kerma_group_coefficients.csv"
)
OUTPUT_DIR = INPUT.parent
GROUPS = (
    "lt0p4MeV",
    "0p4to1MeV",
    "1to5MeV",
    "5to10MeV",
    "10to20MeV",
)
DISPLAY = {
    "lt0p4MeV": "<0,4",
    "0p4to1MeV": "0,4–1",
    "1to5MeV": "1–5",
    "5to10MeV": "5–10",
    "10to20MeV": "10–20",
}
SOURCE_MEAN_MEV = 14.7
SOURCE_SIGMA_MEV = 0.15


def load_rows() -> list[dict[str, str]]:
    with INPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarise(
    rows: list[dict[str, str]],
) -> list[dict[str, float | str]]:
    output: list[dict[str, float | str]] = []
    total_track = sum(
        float(row["gtv_track_length_mm"])
        for row in rows
        if row["energy_group"] in GROUPS
    )
    total_kerma = sum(
        float(row["gtv_water_kerma_Gy_per_primary"])
        for row in rows
        if row["energy_group"] in GROUPS
    )
    for group in GROUPS:
        selected = [row for row in rows if row["energy_group"] == group]
        lengths = np.asarray(
            [float(row["gtv_track_length_mm"]) for row in selected]
        )
        energies = np.asarray(
            [float(row["mean_energy_MeV"]) for row in selected]
        )
        kerma = np.asarray(
            [
                float(row["gtv_water_kerma_Gy_per_primary"])
                for row in selected
            ]
        )
        seed_track_fraction = np.asarray(
            [
                100.0
                * float(row["gtv_track_length_mm"])
                / sum(
                    float(candidate["gtv_track_length_mm"])
                    for candidate in rows
                    if candidate["run_tag"] == row["run_tag"]
                    and candidate["energy_group"] in GROUPS
                )
                for row in selected
            ]
        )
        seed_kerma_fraction = np.asarray(
            [
                100.0
                * float(row["gtv_water_kerma_Gy_per_primary"])
                / sum(
                    float(candidate["gtv_water_kerma_Gy_per_primary"])
                    for candidate in rows
                    if candidate["run_tag"] == row["run_tag"]
                    and candidate["energy_group"] in GROUPS
                )
                for row in selected
            ]
        )
        output.append(
            {
                "energy_group": group,
                "track_weighted_mean_energy_MeV": float(
                    np.average(energies, weights=lengths)
                ),
                "track_length_fraction_percent": (
                    100.0 * float(np.sum(lengths)) / total_track
                ),
                "track_length_seed_min_percent": float(
                    np.min(seed_track_fraction)
                ),
                "track_length_seed_max_percent": float(
                    np.max(seed_track_fraction)
                ),
                "water_kerma_fraction_percent": (
                    100.0 * float(np.sum(kerma)) / total_kerma
                ),
                "water_kerma_seed_min_percent": float(
                    np.min(seed_kerma_fraction)
                ),
                "water_kerma_seed_max_percent": float(
                    np.max(seed_kerma_fraction)
                ),
            }
        )
    return output


def write_summary(rows: list[dict[str, float | str]]) -> None:
    path = OUTPUT_DIR / "neutron_gtv_spectrum_summary.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float | str]]) -> None:
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.8, 7.0),
        gridspec_kw={"width_ratios": (1.0, 1.45)},
    )

    energy = np.linspace(13.9, 15.5, 801)
    density = (
        np.exp(
            -0.5
            * ((energy - SOURCE_MEAN_MEV) / SOURCE_SIGMA_MEV) ** 2
        )
        / (SOURCE_SIGMA_MEV * np.sqrt(2.0 * np.pi))
    )
    axes[0].plot(
        energy,
        density,
        color="#276893",
        linewidth=2.8,
        label="заданный источник",
    )
    axes[0].fill_between(
        energy,
        density,
        where=(
            (energy >= SOURCE_MEAN_MEV - SOURCE_SIGMA_MEV)
            & (energy <= SOURCE_MEAN_MEV + SOURCE_SIGMA_MEV)
        ),
        color="#4d94bf",
        alpha=0.25,
        label="±1σ",
    )
    axes[0].axvline(
        SOURCE_MEAN_MEV,
        color="#173f5f",
        linewidth=1.3,
        linestyle="--",
    )
    axes[0].annotate(
        "14,7 МэВ",
        xy=(SOURCE_MEAN_MEV, density.max()),
        xytext=(SOURCE_MEAN_MEV + 0.08, density.max() * 0.91),
        arrowprops={"arrowstyle": "-", "color": "#173f5f"},
        ha="left",
        va="top",
    )
    axes[0].set_xlabel("энергия первичного нейтрона, МэВ")
    axes[0].set_ylabel("плотность вероятности, МэВ⁻¹")
    axes[0].set_title("Первичный пучок")
    axes[0].set_xlim(13.9, 15.5)
    axes[0].set_ylim(0.0, density.max() * 1.12)
    axes[0].grid(alpha=0.20)
    axes[0].legend(loc="upper left")

    positions = np.arange(len(rows), dtype=float)
    width = 0.36
    track = np.asarray(
        [float(row["track_length_fraction_percent"]) for row in rows]
    )
    kerma = np.asarray(
        [float(row["water_kerma_fraction_percent"]) for row in rows]
    )
    track_min = np.asarray(
        [float(row["track_length_seed_min_percent"]) for row in rows]
    )
    track_max = np.asarray(
        [float(row["track_length_seed_max_percent"]) for row in rows]
    )
    kerma_min = np.asarray(
        [float(row["water_kerma_seed_min_percent"]) for row in rows]
    )
    kerma_max = np.asarray(
        [float(row["water_kerma_seed_max_percent"]) for row in rows]
    )
    track_bars = axes[1].bar(
        positions - width / 2.0,
        track,
        width,
        color="#5f88b5",
        label="трековая длина (флюенс)",
        yerr=np.vstack((track - track_min, track_max - track)),
        capsize=3,
    )
    kerma_bars = axes[1].bar(
        positions + width / 2.0,
        kerma,
        width,
        color="#e68613",
        label="водоэквивалентная керма",
        yerr=np.vstack((kerma - kerma_min, kerma_max - kerma)),
        capsize=3,
    )
    for bars, values in ((track_bars, track), (kerma_bars, kerma)):
        for bar, value in zip(bars, values, strict=True):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                value * 1.18,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                rotation=90 if value < 0.3 else 0,
            )
    labels = [
        f"{DISPLAY[str(row['energy_group'])]}\n"
        f"Ē={float(row['track_weighted_mean_energy_MeV']):.2f}"
        for row in rows
    ]
    axes[1].set_xticks(positions, labels)
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.035, 180.0)
    axes[1].set_ylabel("доля внутри GTV, % (логарифмическая шкала)")
    axes[1].set_xlabel("энергетическая группа, МэВ")
    axes[1].set_title("Спектр после транспорта в GTV")
    axes[1].grid(axis="y", which="both", alpha=0.20)
    axes[1].legend(loc="upper left")
    axes[1].text(
        0.99,
        0.03,
        "Группа ≥20 МэВ: 0%",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
    )

    fig.suptitle(
        "Нейтроны НГ-14: заданный пучок и энергетический состав в GTV",
        fontsize=21,
        y=0.99,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.16,
        top=0.86,
        wspace=0.28,
    )
    fig.savefig(
        OUTPUT_DIR / "neutron_gtv_spectrum.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_DIR / "neutron_gtv_spectrum.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    configurator.restore_original_styles()


def main() -> None:
    rows = summarise(load_rows())
    write_summary(rows)
    plot(rows)


if __name__ == "__main__":
    main()
