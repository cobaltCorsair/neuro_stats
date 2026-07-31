"""Reconstruct an effective post-filter C-12 source from water basis curves.

This is a design reconstruction, not an inverse reconstruction of the lost
physical ridge filter.  Eight archived residual energies are transported
separately through water.  Their non-negative source weights are then adjusted
to make the physical depth-dose plateau flatter, with a ridge penalty towards
the archived weights to avoid exact interpolation of eight noisy depth bins by
eight free weights.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear


ARCHIVED_WEIGHTS = np.asarray(
    [0.20, 0.37, 0.45, 0.54, 0.66, 0.85, 1.00, 2.00],
    dtype=float,
)


def load_basis(
    root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    entries: list[tuple[float, Path, dict]] = []
    for directory in root.glob("E*MeV"):
        metadata_path = directory / "run_metadata.json"
        profile_path = directory / "depth_profile_central_r10mm.csv"
        if not metadata_path.exists() or not profile_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        entries.append(
            (
                float(metadata["total_kinetic_energy_MeV"]),
                profile_path,
                metadata,
            )
        )
    entries.sort(key=lambda item: item[0])
    if len(entries) != 8:
        raise RuntimeError(
            f"Expected eight complete basis responses, found {len(entries)}"
        )

    depth: np.ndarray | None = None
    curves: list[np.ndarray] = []
    metadata_rows: list[dict] = []
    for energy, profile_path, metadata in entries:
        with profile_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        current_depth = np.asarray(
            [float(row["depth_mm"]) for row in rows],
            dtype=float,
        )
        histories = int(metadata["histories"])
        curve = np.asarray(
            [float(row["depEnergy_keV"]) for row in rows],
            dtype=float,
        ) / histories
        if depth is None:
            depth = current_depth
        elif not np.array_equal(depth, current_depth):
            raise RuntimeError(f"Depth grid mismatch in {profile_path}")
        curves.append(curve)
        metadata_rows.append(metadata)

    assert depth is not None
    energies = np.asarray([item[0] for item in entries], dtype=float)
    return depth, energies, np.column_stack(curves), metadata_rows


def plateau_metrics(values: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    return {
        "mean_keV_per_primary_relative_fluence": mean,
        "sd_keV_per_primary_relative_fluence": standard_deviation,
        "coefficient_of_variation": standard_deviation / mean,
        "minimum_fraction_of_mean": float(np.min(values) / mean),
        "maximum_fraction_of_mean": float(np.max(values) / mean),
        "peak_to_peak_fraction_of_mean": float(
            (np.max(values) - np.min(values)) / mean
        ),
    }


def write_effective_macro(
    path: Path,
    energies: np.ndarray,
    weights: np.ndarray,
    histories: int = 10000,
) -> None:
    lines = [
        "# Computationally reconstructed effective post-filter C-12 source.",
        "# This is not a geometric model of the lost historical filter.",
        "# Energies are total kinetic energies per C-12 ion.",
        "",
        "/control/verbose 0",
        "/run/verbose 1",
        "/event/verbose 0",
        "/tracking/verbose 0",
        "/run/printProgress 500",
        "/random/setSeeds 44119 77213",
        "",
        "/gps/source/clear",
        "",
    ]
    for energy, weight in zip(energies, weights, strict=True):
        lines.extend(
            [
                f"/gps/source/add {weight:.8f}",
                "/gps/particle ion",
                "/gps/ion 6 12 6 0",
                "/gps/ene/type Mono",
                f"/gps/ene/mono {energy:.1f} MeV",
                "/gps/pos/type Plane",
                "/gps/pos/shape Rectangle",
                "/gps/pos/centre -180.1 0.0 0.0 mm",
                "/gps/pos/rot1 0.0 1.0 0.0",
                "/gps/pos/rot2 0.0 0.0 1.0",
                "/gps/pos/halfx 15.0 mm",
                "/gps/pos/halfy 15.0 mm",
                "/gps/direction 1.0 0.0 0.0",
                "",
            ]
        )
    lines.append(f"/run/beamOn {histories}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(
    output_path: Path,
    depth: np.ndarray,
    energies: np.ndarray,
    basis: np.ndarray,
    plateau_mask: np.ndarray,
    archive_profile: np.ndarray,
    optimised_profile: np.ndarray,
    archived_weights: np.ndarray,
    optimised_weights: np.ndarray,
) -> None:
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
        3,
        1,
        figsize=(9.0, 9.6),
        gridspec_kw={"height_ratios": [1.1, 1.25, 0.85]},
        constrained_layout=True,
    )
    show = depth <= 24.5
    palette = plt.cm.viridis(np.linspace(0.08, 0.92, len(energies)))

    for column, (energy, colour) in enumerate(
        zip(energies, palette, strict=True)
    ):
        curve = basis[:, column]
        axes[0].plot(
            depth[show],
            curve[show] / np.max(curve),
            color=colour,
            linewidth=1.4,
            label=f"{energy / 12:.1f} МэВ/нуклон",
        )
    axes[0].set(
        ylabel="Относительный\nэнерговклад",
        title="Моноэнергетические базисные кривые в воде",
        xlim=(0, 24.5),
        ylim=(0, 1.08),
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(ncol=4, frameon=False, loc="upper left")

    plateau_depths = depth[plateau_mask]
    for profile, label, colour, style in (
        (
            archive_profile,
            "архивные веса",
            "#737373",
            "--",
        ),
        (
            optimised_profile,
            "регуляризованная оптимизация",
            "#7a1fa2",
            "-",
        ),
    ):
        normalised = profile / np.mean(profile[plateau_mask])
        axes[1].plot(
            depth[show],
            normalised[show],
            color=colour,
            linestyle=style,
            linewidth=2.2,
            label=label,
        )
    axes[1].axvspan(
        plateau_depths[0] - 0.5,
        plateau_depths[-1] + 0.5,
        color="#f0a202",
        alpha=0.13,
        label="расчётное плато",
    )
    axes[1].axhline(1.0, color="#444444", linewidth=0.8)
    axes[1].set(
        ylabel="Энерговклад / среднее\nна плато",
        title="Суммарный физический профиль",
        xlim=(0, 24.5),
        ylim=(0, 1.22),
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, loc="upper left")

    positions = np.arange(len(energies))
    width = 0.38
    axes[2].bar(
        positions - width / 2,
        archived_weights,
        width,
        color="#9c9c9c",
        label="архивные",
    )
    axes[2].bar(
        positions + width / 2,
        optimised_weights,
        width,
        color="#7a1fa2",
        label="оптимизированные",
    )
    axes[2].set_xticks(
        positions,
        [f"{energy / 12:.1f}" for energy in energies],
    )
    axes[2].set(
        xlabel="Энергия, МэВ/нуклон",
        ylabel="Относительный вес\n(max = 1)",
        title="Спектральные веса эффективного источника",
        ylim=(0, 1.12),
    )
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].legend(frameon=False, loc="upper left")

    figure.suptitle(
        "Расчётное восстановление эффективного источника после фильтра",
        fontsize=13,
    )
    for suffix in (".png", ".svg", ".pdf"):
        figure.savefig(output_path.with_suffix(suffix), dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    depth, energies, basis, metadata = load_basis(args.basis_root)
    if len(ARCHIVED_WEIGHTS) != len(energies):
        raise RuntimeError("Archived weight count does not match basis count")

    # Peak search is restricted to the physically relevant shallow region so
    # isolated fragment deposits far downstream cannot define the range.
    peak_search = depth <= 30.0
    peak_indices = np.asarray(
        [
            np.flatnonzero(peak_search)[np.argmax(basis[peak_search, column])]
            for column in range(basis.shape[1])
        ],
        dtype=int,
    )
    peak_depths = depth[peak_indices]
    plateau_start = float(np.min(peak_depths))
    plateau_end = float(np.max(peak_depths))
    plateau_mask = (depth >= plateau_start) & (depth <= plateau_end)

    archived_relative = ARCHIVED_WEIGHTS / np.max(ARCHIVED_WEIGHTS)
    design = basis[plateau_mask]
    response_scale = float(np.max(design))
    design_scaled = design / response_scale
    ridge = float(args.ridge_lambda)
    augmented_design = np.vstack(
        [design_scaled, np.sqrt(ridge) * np.eye(len(energies))]
    )
    augmented_target = np.concatenate(
        [
            np.ones(np.count_nonzero(plateau_mask), dtype=float),
            np.sqrt(ridge) * archived_relative,
        ]
    )
    result = lsq_linear(
        augmented_design,
        augmented_target,
        bounds=(0.0, np.inf),
        method="trf",
        lsmr_tol="auto",
    )
    if not result.success:
        raise RuntimeError(f"Non-negative optimisation failed: {result.message}")
    optimised_relative = result.x / np.max(result.x)

    archive_profile = basis @ archived_relative
    optimised_profile = basis @ optimised_relative
    archive_metrics = plateau_metrics(archive_profile[plateau_mask])
    optimised_metrics = plateau_metrics(optimised_profile[plateau_mask])

    weights_path = args.output_dir / "effective_source_weights.csv"
    with weights_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "total_energy_MeV",
                "energy_per_nucleon_MeV",
                "basis_peak_depth_mm",
                "archived_weight",
                "archived_weight_max_normalised",
                "optimised_weight_max_normalised",
            ]
        )
        for energy, peak, archived, archived_norm, optimised in zip(
            energies,
            peak_depths,
            ARCHIVED_WEIGHTS,
            archived_relative,
            optimised_relative,
            strict=True,
        ):
            writer.writerow(
                [
                    f"{energy:.1f}",
                    f"{energy / 12.0:.6f}",
                    f"{peak:.3f}",
                    f"{archived:.8f}",
                    f"{archived_norm:.8f}",
                    f"{optimised:.8f}",
                ]
            )

    profiles_path = args.output_dir / "predicted_depth_profiles.csv"
    with profiles_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "depth_mm",
                "in_optimisation_plateau",
                "archived_profile_keV_per_primary_relative_fluence",
                "optimised_profile_keV_per_primary_relative_fluence",
                "archived_profile_normalised_to_plateau_mean",
                "optimised_profile_normalised_to_plateau_mean",
            ]
        )
        archive_mean = np.mean(archive_profile[plateau_mask])
        optimised_mean = np.mean(optimised_profile[plateau_mask])
        for row in zip(
            depth,
            plateau_mask,
            archive_profile,
            optimised_profile,
            archive_profile / archive_mean,
            optimised_profile / optimised_mean,
            strict=True,
        ):
            writer.writerow(
                [
                    f"{row[0]:.3f}",
                    int(row[1]),
                    f"{row[2]:.10g}",
                    f"{row[3]:.10g}",
                    f"{row[4]:.10g}",
                    f"{row[5]:.10g}",
                ]
            )

    macro_path = args.output_dir / "carbon_c12_effective_postfilter_water.mac"
    write_effective_macro(macro_path, energies, optimised_relative)
    make_plot(
        args.output_dir / "carbon_c12_postfilter_reconstruction.png",
        depth,
        energies,
        basis,
        plateau_mask,
        archive_profile,
        optimised_profile,
        archived_relative,
        optimised_relative,
    )

    summary = {
        "status": "computational_effective_source_not_historical_filter",
        "basis_root": str(args.basis_root),
        "physics": metadata[0]["physics"],
        "histories_per_basis_energy": metadata[0]["histories"],
        "ridge_lambda": ridge,
        "optimisation": (
            "non-negative least squares with Tikhonov penalty towards "
            "archived weights"
        ),
        "plateau_start_mm": plateau_start,
        "plateau_end_mm": plateau_end,
        "plateau_width_mm": plateau_end - plateau_start,
        "basis_peak_depths_mm": peak_depths.tolist(),
        "archived_metrics": archive_metrics,
        "optimised_metrics": optimised_metrics,
        "coefficient_of_variation_relative_reduction": (
            1.0
            - optimised_metrics["coefficient_of_variation"]
            / archive_metrics["coefficient_of_variation"]
        ),
        "limitations": [
            "The lost filter shape and material are not reconstructed.",
            "The plateau is a computational design objective in water.",
            "The weights require an independent mixed-source Monte Carlo run.",
            "No claim of historical series-specific dosimetric identity is made.",
        ],
    }
    (args.output_dir / "optimisation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
