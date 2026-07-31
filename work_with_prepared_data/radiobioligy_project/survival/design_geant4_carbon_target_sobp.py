"""Design a target-width C-12 effective source from water basis curves.

The output is a prospective source model for a selected ROI width.  It is not
an inverse reconstruction of the unavailable historical ridge filter.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear


def load_basis(
    roots: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    entries: list[tuple[float, Path, dict]] = []
    for root in roots:
        for directory in root.glob("E*MeV"):
            metadata_path = directory / "run_metadata.json"
            profile_path = directory / "depth_profile_central_r10mm.csv"
            if metadata_path.exists() and profile_path.exists():
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8-sig")
                )
                entries.append(
                    (
                        float(metadata["total_kinetic_energy_MeV"]),
                        profile_path,
                        metadata,
                    )
                )
    entries.sort(key=lambda item: item[0])
    if len(entries) < 3:
        raise RuntimeError("At least three complete basis responses are required")
    energies_found = [entry[0] for entry in entries]
    if len(set(energies_found)) != len(energies_found):
        raise RuntimeError("Duplicate basis energies found across --basis-root inputs")

    depth: np.ndarray | None = None
    curves: list[np.ndarray] = []
    metadata_rows: list[dict] = []
    for _, profile_path, metadata in entries:
        with profile_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        current_depth = np.asarray(
            [float(row["depth_mm"]) for row in rows],
            dtype=float,
        )
        curve = np.asarray(
            [float(row["depEnergy_keV"]) for row in rows],
            dtype=float,
        ) / int(metadata["histories"])
        if depth is None:
            depth = current_depth
        elif not np.array_equal(depth, current_depth):
            raise RuntimeError(f"Depth grid mismatch in {profile_path}")
        curves.append(curve)
        metadata_rows.append(metadata)

    assert depth is not None
    energies = np.asarray([item[0] for item in entries], dtype=float)
    return depth, energies, np.column_stack(curves), metadata_rows


def second_difference_matrix(size: int) -> np.ndarray:
    matrix = np.zeros((size - 2, size), dtype=float)
    for index in range(size - 2):
        matrix[index, index : index + 3] = (1.0, -2.0, 1.0)
    return matrix


def plateau_metrics(values: np.ndarray) -> dict[str, float]:
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


def write_macro(
    path: Path,
    energies: np.ndarray,
    weights: np.ndarray,
    histories: int = 20000,
) -> None:
    lines = [
        "# Prospective target-width C-12 effective source.",
        "# This source is not a reconstruction of the historical ridge filter.",
        "# Energies are total kinetic energies per C-12 ion.",
        "",
        "/control/verbose 0",
        "/run/verbose 1",
        "/event/verbose 0",
        "/tracking/verbose 0",
        "/run/printProgress 1000",
        "/random/setSeeds 61283 90841",
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
    output: Path,
    depth: np.ndarray,
    energies: np.ndarray,
    basis: np.ndarray,
    plateau: np.ndarray,
    profile: np.ndarray,
    weights: np.ndarray,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
        }
    )
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(9.0, 9.5),
        gridspec_kw={"height_ratios": [1.05, 1.15, 0.8]},
        constrained_layout=True,
    )
    show = depth <= 28.5
    colours = plt.cm.viridis(np.linspace(0.05, 0.95, len(energies)))
    for column, (energy, colour) in enumerate(
        zip(energies, colours, strict=True)
    ):
        curve = basis[:, column]
        axes[0].plot(
            depth[show],
            curve[show] / np.max(curve),
            color=colour,
            linewidth=1.05,
        )
    axes[0].set(
        title="Расширенный моноэнергетический базис в воде",
        ylabel="Относительный\nэнерговклад",
        xlim=(0, 28.5),
        ylim=(0, 1.08),
    )
    axes[0].grid(alpha=0.2)

    plateau_depth = depth[plateau]
    profile_normalised = profile / np.mean(profile[plateau])
    axes[1].axvspan(
        plateau_depth[0] - 0.5,
        plateau_depth[-1] + 0.5,
        color="#f0a202",
        alpha=0.13,
        label="проектное плато",
    )
    axes[1].plot(
        depth[show],
        profile_normalised[show],
        color="#7a1fa2",
        linewidth=2.3,
        marker="o",
        markersize=3.0,
        label="предсказанный смешанный профиль",
    )
    axes[1].axhline(1.0, color="#444444", linewidth=0.8)
    axes[1].set(
        title="Проектный физический профиль",
        ylabel="Энерговклад / среднее\nна плато",
        xlim=(0, 28.5),
        ylim=(0, 1.18),
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, loc="upper left")

    positions = np.arange(len(energies))
    axes[2].bar(positions, weights, color="#7a1fa2", width=0.72)
    axes[2].set_xticks(
        positions,
        [f"{energy / 12:.0f}" for energy in energies],
        rotation=45,
        ha="right",
    )
    axes[2].set(
        title="Веса проектного эффективного источника",
        xlabel="Энергия, МэВ/нуклон",
        ylabel="Относительный вес\n(max = 1)",
        ylim=(0, 1.08),
    )
    axes[2].grid(axis="y", alpha=0.2)

    figure.suptitle(
        "Проектный источник углеродного пика для ширины GTVp",
        fontsize=13,
    )
    for suffix in (".png", ".svg", ".pdf"):
        figure.savefig(output.with_suffix(suffix), dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--basis-root",
        type=Path,
        action="append",
        required=True,
        help="Basis directory; may be specified more than once.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoothness-lambda", type=float, default=0.1)
    parser.add_argument("--ridge-lambda", type=float, default=0.01)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    depth, energies, basis, metadata = load_basis(args.basis_root)
    peak_search = depth <= 40.0
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
    plateau = (depth >= plateau_start) & (depth <= plateau_end)
    design = basis[plateau]
    scale = float(np.max(design))

    smoothness = second_difference_matrix(len(energies))
    augmented_design = np.vstack(
        [
            design / scale,
            np.sqrt(args.smoothness_lambda) * smoothness,
            np.sqrt(args.ridge_lambda) * np.eye(len(energies)),
        ]
    )
    augmented_target = np.concatenate(
        [
            np.ones(np.count_nonzero(plateau), dtype=float),
            np.zeros(smoothness.shape[0], dtype=float),
            np.zeros(len(energies), dtype=float),
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
        raise RuntimeError(result.message)
    weights = result.x / np.max(result.x)
    profile = basis @ weights
    fit_metrics = plateau_metrics(profile[plateau])

    with (args.output_dir / "effective_source_weights.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "total_energy_MeV",
                "energy_per_nucleon_MeV",
                "basis_peak_depth_mm",
                "optimised_weight_max_normalised",
            ]
        )
        for energy, peak, weight in zip(
            energies, peak_depths, weights, strict=True
        ):
            writer.writerow(
                [
                    f"{energy:.1f}",
                    f"{energy / 12:.6f}",
                    f"{peak:.3f}",
                    f"{weight:.8f}",
                ]
            )

    mean_profile = float(np.mean(profile[plateau]))
    with (args.output_dir / "predicted_depth_profiles.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "depth_mm",
                "in_optimisation_plateau",
                "optimised_profile_keV_per_primary_relative_fluence",
                "optimised_profile_normalised_to_plateau_mean",
            ]
        )
        for row in zip(
            depth,
            plateau,
            profile,
            profile / mean_profile,
            strict=True,
        ):
            writer.writerow(
                [
                    f"{row[0]:.3f}",
                    int(row[1]),
                    f"{row[2]:.10g}",
                    f"{row[3]:.10g}",
                ]
            )

    write_macro(
        args.output_dir / "carbon_c12_effective_postfilter_water.mac",
        energies,
        weights,
    )
    make_plot(
        args.output_dir / "carbon_c12_target_width_design.png",
        depth,
        energies,
        basis,
        plateau,
        profile,
        weights,
    )
    summary = {
        "status": "prospective_target_width_effective_source",
        "historical_filter_reconstruction": False,
        "basis_roots": [str(path) for path in args.basis_root],
        "physics": metadata[0]["physics"],
        "histories_per_basis_energy": metadata[0]["histories"],
        "number_of_energy_components": len(energies),
        "smoothness_lambda": args.smoothness_lambda,
        "ridge_lambda": args.ridge_lambda,
        "plateau_start_mm": plateau_start,
        "plateau_end_mm": plateau_end,
        "plateau_width_mm": plateau_end - plateau_start,
        "basis_peak_depths_mm": peak_depths.tolist(),
        "predicted_metrics": fit_metrics,
        "limitations": [
            "Prospective source design, not a historical filter reconstruction.",
            "Pure primary C-12 GPS has no filter-generated fragments.",
            "Water optimisation must be checked by a direct mixed run.",
            "Rat placement and GTV DVH require a separate calculation.",
        ],
    }
    (args.output_dir / "optimisation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
