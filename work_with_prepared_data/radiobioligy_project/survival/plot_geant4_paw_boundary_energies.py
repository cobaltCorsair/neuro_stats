"""Summarise primary-proton energy at the entrance and exit of the rat paw.

The input CSV is produced by the optional Geant4 stepping scorer in
``app2dlls``.  Tissue is defined by a configurable material-density threshold.
An exit energy is reported only when the original primary proton leaves the
contiguous tissue segment in the forward direction.  Events without such a
crossing are retained as non-transmitted primaries, not silently discarded.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (
    MatplotlibConfigurator,
)


DEFAULT_INPUT = Path(
    r"C:\dev\dissertation\task4_5"
) / "scoring_v2_proton_40MeV_L_to_R_paw_boundary_1k" / (
    "paw_boundary_crossings.csv"
)
DEFAULT_OUTPUT = Path(
    r"C:\dev\dissertation\task4_5\outputs\geant4_livermore_20260724"
) / "proton_paw_boundary_energy"


def describe_energy(
    values: pd.Series,
    group: str,
) -> dict[str, float | int | str]:
    array = values.dropna().to_numpy(dtype=float)
    if not len(array):
        return {
            "group": group,
            "n": 0,
            "minimum_MeV": np.nan,
            "p05_MeV": np.nan,
            "median_MeV": np.nan,
            "mean_MeV": np.nan,
            "p95_MeV": np.nan,
            "maximum_MeV": np.nan,
            "sd_MeV": np.nan,
        }
    return {
        "group": group,
        "n": len(array),
        "minimum_MeV": float(np.min(array)),
        "p05_MeV": float(np.quantile(array, 0.05)),
        "median_MeV": float(np.median(array)),
        "mean_MeV": float(np.mean(array)),
        "p95_MeV": float(np.quantile(array, 0.95)),
        "maximum_MeV": float(np.max(array)),
        "sd_MeV": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }


def paired_crossings(
    crossings: pd.DataFrame,
    histories: int | None,
) -> tuple[pd.DataFrame, int]:
    crossings = crossings.sort_values(
        ["event_id", "crossing_index"],
    )
    entries = (
        crossings.loc[crossings["boundary"] == "entry"]
        .drop_duplicates("event_id", keep="first")
        .set_index("event_id")
        .add_prefix("entry_")
    )
    final_crossings = (
        crossings.drop_duplicates("event_id", keep="last")
        .set_index("event_id")
    )
    exits = (
        final_crossings.loc[final_crossings["boundary"] == "exit"]
        .add_prefix("exit_")
    )
    paired = entries.join(exits, how="left")
    paired["transmitted"] = paired["exit_kinetic_energy_MeV"].notna()
    paired["exit_energy_with_stops_MeV"] = (
        paired["exit_kinetic_energy_MeV"].fillna(0.0)
    )
    paired["energy_loss_MeV"] = (
        paired["entry_kinetic_energy_MeV"]
        - paired["exit_energy_with_stops_MeV"]
    )
    paired["entry_radius_mm"] = np.hypot(
        paired["entry_x_mm"],
        paired["entry_z_mm"],
    )
    paired["path_length_mm"] = np.sqrt(
        (paired["exit_x_mm"] - paired["entry_x_mm"]) ** 2
        + (paired["exit_y_mm"] - paired["entry_y_mm"]) ** 2
        + (paired["exit_z_mm"] - paired["entry_z_mm"]) ** 2
    )
    paired["boundary_crossing_count"] = crossings.groupby(
        "event_id",
    ).size()
    if histories is None:
        histories = (
            int(crossings["event_id"].max()) + 1 if len(crossings) else 0
        )
    return paired, histories


def write_outputs(
    paired: pd.DataFrame,
    histories: int,
    output_dir: Path,
    core_radius_mm: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    transmitted = paired["transmitted"]
    core = paired["entry_radius_mm"] <= core_radius_mm
    summary_rows = [
        describe_energy(
            paired["entry_kinetic_energy_MeV"],
            "entry_all_tissue_crossing_primaries",
        ),
        describe_energy(
            paired.loc[transmitted, "exit_kinetic_energy_MeV"],
            "exit_transmitted_primaries_only",
        ),
        describe_energy(
            paired["exit_energy_with_stops_MeV"],
            "exit_with_nontransmitted_as_zero",
        ),
        describe_energy(
            paired.loc[core, "entry_kinetic_energy_MeV"],
            f"entry_central_core_r_le_{core_radius_mm:g}_mm",
        ),
        describe_energy(
            paired.loc[core & transmitted, "exit_kinetic_energy_MeV"],
            f"exit_central_core_transmitted_r_le_{core_radius_mm:g}_mm",
        ),
    ]
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "paw_boundary_energy_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.reset_index().to_csv(
        output_dir / "paw_boundary_energy_paired_events.csv",
        index=False,
        encoding="utf-8-sig",
    )

    n_entry = len(paired)
    n_exit = int(transmitted.sum())
    n_core = int(core.sum())
    n_core_exit = int((core & transmitted).sum())
    with (
        output_dir / "paw_boundary_transmission_summary.csv"
    ).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["simulated_histories", histories])
        writer.writerow(["primaries_entering_tissue", n_entry])
        writer.writerow(["primaries_exiting_tissue", n_exit])
        writer.writerow(
            [
                "entry_fraction_of_all_histories",
                n_entry / histories if histories else np.nan,
            ]
        )
        writer.writerow(
            [
                "transmission_fraction_among_entries",
                n_exit / n_entry if n_entry else np.nan,
            ]
        )
        writer.writerow(["central_core_radius_mm", core_radius_mm])
        writer.writerow(["central_core_entries", n_core])
        writer.writerow(["central_core_exits", n_core_exit])
        writer.writerow(
            [
                "central_core_transmission_fraction",
                n_core_exit / n_core if n_core else np.nan,
            ]
        )


def plot_summary(
    paired: pd.DataFrame,
    output_dir: Path,
) -> None:
    transmitted = paired["transmitted"]
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
            }
        )
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(13.5, 6.0),
            constrained_layout=False,
        )
        fig.subplots_adjust(
            left=0.075,
            right=0.925,
            bottom=0.12,
            top=0.79,
            wspace=0.16,
        )
        entry = paired["entry_kinetic_energy_MeV"].to_numpy(dtype=float)
        exit_energy = paired.loc[
            transmitted,
            "exit_kinetic_energy_MeV",
        ].to_numpy(dtype=float)
        bins = np.linspace(
            0.0,
            max(40.5, float(np.nanmax(entry)) + 0.5),
            82,
        )
        axes[0].hist(
            entry,
            bins=bins,
            weights=np.full(len(entry), 100.0 / len(entry)),
            color="#377eb8",
            alpha=0.72,
            label=f"вход, n={len(entry)}",
        )
        axes[0].hist(
            exit_energy,
            bins=bins,
            weights=np.full(
                len(exit_energy),
                100.0 / len(exit_energy),
            ),
            color="#e6550d",
            alpha=0.68,
            label=f"выход, n={len(exit_energy)}",
        )
        axes[0].set(
            xlabel="кинетическая энергия первичного протона, МэВ",
            ylabel="доля внутри группы, %",
            title="Энергия на границах опухолевой лапы",
        )
        axes[0].grid(alpha=0.18)
        axes[0].legend()
        stopped_fraction = 100.0 * (1.0 - len(exit_energy) / len(entry))
        axes[0].text(
            0.03,
            0.78,
            (
                f"нет дистального выхода: {stopped_fraction:.1f}%\n"
                "для них остаточная энергия условно равна 0"
            ),
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            bbox={
                "facecolor": "white",
                "edgecolor": "#bbbbbb",
                "alpha": 0.88,
            },
        )

        scatter = paired.loc[
            transmitted & paired["path_length_mm"].notna()
        ]
        axes[1].scatter(
            scatter["path_length_mm"],
            scatter["exit_kinetic_energy_MeV"],
            c=scatter["entry_radius_mm"],
            cmap="viridis",
            s=14,
            alpha=0.58,
            edgecolors="none",
        )
        axes[1].set(
            xlabel="геометрическая длина пути между границами, мм",
            ylabel="энергия при выходе, МэВ",
            title="Остаточная энергия прошедших протонов",
        )
        axes[1].grid(alpha=0.18)
        colorbar = fig.colorbar(
            axes[1].collections[0],
            ax=axes[1],
            fraction=0.05,
            pad=0.03,
        )
        colorbar.set_label("радиус точки входа относительно оси, мм")
        fig.suptitle(
            "Протоны 40 МэВ, L→R; QGSP_INCLXX + G4EmLivermore",
            fontsize=14,
            y=0.94,
        )
        stem = output_dir / "paw_boundary_energy_distributions"
        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(
                stem.with_suffix(suffix),
                dpi=300 if suffix == ".png" else None,
                bbox_inches="tight",
                pad_inches=0.25,
            )
        plt.close(fig)
    finally:
        configurator.restore_original_styles()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--histories", type=int, default=1000)
    parser.add_argument("--core-radius-mm", type=float, default=5.0)
    args = parser.parse_args()

    crossings = pd.read_csv(args.input)
    paired, histories = paired_crossings(crossings, args.histories)
    write_outputs(
        paired,
        histories,
        args.output_dir,
        args.core_radius_mm,
    )
    plot_summary(paired, args.output_dir)

    entry = paired["entry_kinetic_energy_MeV"]
    exit_energy = paired.loc[
        paired["transmitted"],
        "exit_kinetic_energy_MeV",
    ]
    print(
        "entry_MeV="
        f"{entry.min():.6g}..{entry.max():.6g}; "
        "exit_transmitted_MeV="
        f"{exit_energy.min():.6g}..{exit_energy.max():.6g}; "
        f"entries={len(entry)}; exits={len(exit_energy)}"
    )


if __name__ == "__main__":
    main()
