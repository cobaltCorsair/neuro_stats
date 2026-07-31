"""Validate a reconstructed mixed C-12 source in an independent direct run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def metrics(values: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    return {
        "mean": mean,
        "sd": sd,
        "coefficient_of_variation": sd / mean,
        "minimum_fraction_of_mean": float(np.min(values) / mean),
        "maximum_fraction_of_mean": float(np.max(values) / mean),
        "peak_to_peak_fraction_of_mean": float(np.ptp(values) / mean),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimisation-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    args = parser.parse_args()

    optimisation = json.loads(
        (args.optimisation_dir / "optimisation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = json.loads(
        (args.validation_dir / "run_metadata.json").read_text(
            encoding="utf-8-sig"
        )
    )
    with (
        args.optimisation_dir / "predicted_depth_profiles.csv"
    ).open(encoding="utf-8") as handle:
        predicted_rows = list(csv.DictReader(handle))
    with (
        args.validation_dir / "depth_profile_central_r10mm.csv"
    ).open(encoding="utf-8") as handle:
        validation_rows = list(csv.DictReader(handle))

    predicted_depth = np.asarray(
        [float(row["depth_mm"]) for row in predicted_rows]
    )
    depth = np.asarray([float(row["depth_mm"]) for row in validation_rows])
    if not np.array_equal(predicted_depth, depth):
        raise RuntimeError("Predicted and direct profiles use different grids")

    predicted = np.asarray(
        [
            float(
                row[
                    "optimised_profile_normalised_to_plateau_mean"
                ]
            )
            for row in predicted_rows
        ]
    )
    deposited = np.asarray(
        [float(row["depEnergy_keV"]) for row in validation_rows]
    )
    let_base = np.asarray(
        [float(row["letBase_keV2_um"]) for row in validation_rows]
    )
    histories = int(metadata["histories"])
    direct = deposited / histories
    letd = np.divide(
        let_base,
        deposited,
        out=np.full_like(let_base, np.nan),
        where=deposited > 0.0,
    )

    start = float(optimisation["plateau_start_mm"])
    end = float(optimisation["plateau_end_mm"])
    plateau = (depth >= start) & (depth <= end)
    direct_normalised = direct / np.mean(direct[plateau])
    residual = direct_normalised[plateau] - predicted[plateau]
    direct_metrics = metrics(direct[plateau])
    direct_metrics["dose_weighted_LET_keV_um"] = float(
        np.sum(let_base[plateau]) / np.sum(deposited[plateau])
    )
    direct_metrics["predicted_vs_direct_RMSE_fraction_of_mean"] = float(
        np.sqrt(np.mean(residual**2))
    )
    direct_metrics["predicted_vs_direct_max_abs_fraction_of_mean"] = float(
        np.max(np.abs(residual))
    )

    validation_summary = {
        "status": "independent_direct_mixed_source_monte_carlo_validation",
        "histories": histories,
        "seed1": metadata["seed1"],
        "seed2": metadata["seed2"],
        "physics": metadata["physics"],
        "plateau_start_mm": start,
        "plateau_end_mm": end,
        "direct_mixed_source_metrics": direct_metrics,
        "interpretation": (
            "The direct GPS mixture reproduces the optimised water plateau. "
            "This validates the effective source implementation, not the "
            "geometry or dosimetry of the lost historical filter."
        ),
    }
    (args.validation_dir / "validation_summary.json").write_text(
        json.dumps(validation_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.8, 6.8),
        sharex=True,
        constrained_layout=True,
    )
    show = depth <= 24.5
    axes[0].axvspan(
        start - 0.5,
        end + 0.5,
        color="#f0a202",
        alpha=0.13,
        label="расчётное плато",
    )
    axes[0].plot(
        depth[show],
        predicted[show],
        color="#777777",
        linestyle="--",
        linewidth=2.0,
        label="линейное предсказание по базису",
    )
    axes[0].plot(
        depth[show],
        direct_normalised[show],
        color="#7a1fa2",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        label=f"прямой смешанный расчёт, n={histories:,}".replace(",", " "),
    )
    axes[0].axhline(1.0, color="#444444", linewidth=0.8)
    axes[0].set(
        ylabel="Энерговклад / среднее\nна плато",
        title="Независимая проверка смешанного эффективного источника",
        xlim=(0, 24.5),
        ylim=(0, 1.22),
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, loc="lower left")

    axes[1].axvspan(
        start - 0.5,
        end + 0.5,
        color="#f0a202",
        alpha=0.13,
    )
    axes[1].plot(
        depth[show],
        letd[show],
        color="#2457a7",
        linewidth=2.0,
    )
    axes[1].axhline(
        direct_metrics["dose_weighted_LET_keV_um"],
        color="#2457a7",
        linestyle="--",
        linewidth=1.2,
        label=(
            "средняя на плато "
            f"{direct_metrics['dose_weighted_LET_keV_um']:.1f} кэВ/мкм"
        ),
    )
    axes[1].set(
        xlabel="Глубина в воде, мм",
        ylabel=r"$LET_D$, кэВ/мкм",
        title="Дозо-взвешенная электронная ЛПЭ",
        xlim=(0, 24.5),
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, loc="upper left")

    for suffix in (".png", ".svg", ".pdf"):
        figure.savefig(
            args.validation_dir
            / f"carbon_c12_effective_postfilter_validation{suffix}",
            dpi=220,
        )
    plt.close(figure)
    print(json.dumps(validation_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
