"""Summarise paired NG-14 energy and field-radius sensitivity runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


NEURO_STATS_ROOT = Path(__file__).resolve().parents[3]
if str(NEURO_STATS_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURO_STATS_ROOT))

from work_with_prepared_data.radiobioligy_project.survival.analyze_geant4_neutron_ng14_rat import (  # noqa: E402
    DEFAULT_OUTPUT,
    DEFAULT_RUN_ROOT,
    analyse_seed,
)
from work_with_prepared_data.radiobioligy_project.survival.analyze_geant4_proton_100MeV_reduced_gtv import (  # noqa: E402
    DEFAULT_CT_DIR,
    DEFAULT_RS,
    gtv_geometry,
)
from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (  # noqa: E402
    MatplotlibConfigurator,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    _, gtv_ids, gtv_lookup, _ = gtv_geometry(
        Path(DEFAULT_CT_DIR),
        Path(DEFAULT_RS),
    )
    run_dirs = sorted(
        directory
        for directory in args.run_root.iterdir()
        if directory.is_dir()
        and directory.name.startswith("sensitivity_")
        and list(directory.glob("vox_*_0"))
    )
    if len(run_dirs) != 3:
        raise RuntimeError(
            f"Expected three complete sensitivity runs, found {len(run_dirs)}"
        )

    rows: list[dict[str, float | int | str]] = []
    for run_dir in run_dirs:
        metadata = json.loads(
            (run_dir / "run_metadata.json").read_text(encoding="utf-8-sig")
        )
        summary, _ = analyse_seed(run_dir, gtv_ids, gtv_lookup)
        radius = float(metadata["source_radius_mm"])
        dmean_per_primary = float(summary["Dmean_Gy_per_incident_neutron"])
        total_energy = float(summary["gtv_depEnergy_keV"])
        hydrogen_energy = float(summary["hydrogen_energy_keV"])
        rows.append(
            {
                "run": run_dir.name,
                "energy_mean_MeV": float(metadata["energy_mean_MeV"]),
                "energy_sigma_MeV": float(metadata["energy_sigma_MeV"]),
                "field_radius_mm": radius,
                "histories": int(metadata["histories"]),
                "Dmean_Gy_per_incident_neutron": dmean_per_primary,
                "Dmean_times_field_area_Gy_mm2_per_neutron": (
                    dmean_per_primary * np.pi * radius**2
                ),
                "LETd_w_all_keV_um": float(summary["LETd_w_all_keV_um"]),
                "LETd_w_hydrogen_keV_um": float(
                    summary["hydrogen_LETd_w_keV_um"]
                ),
                "hydrogen_dose_fraction_percent": (
                    100.0 * hydrogen_energy / total_energy
                ),
            }
        )

    baseline = next(row for row in rows if "baseline" in str(row["run"]))
    comparison_keys = (
        "Dmean_Gy_per_incident_neutron",
        "Dmean_times_field_area_Gy_mm2_per_neutron",
        "LETd_w_all_keV_um",
        "LETd_w_hydrogen_keV_um",
        "hydrogen_dose_fraction_percent",
    )
    for row in rows:
        for key in comparison_keys:
            row[f"{key}_relative_to_baseline_percent"] = (
                100.0 * float(row[key]) / float(baseline[key]) - 100.0
            )

    csv_path = args.output / "neutron_ng14_source_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    order = [
        next(row for row in rows if "baseline" in str(row["run"])),
        next(row for row in rows if "energy_14p1" in str(row["run"])),
        next(row for row in rows if "aperture" in str(row["run"])),
    ]
    labels = [
        "14,7 МэВ\nr=17,3 мм",
        "14,1 МэВ\nr=17,3 мм",
        "14,7 МэВ\nr=14,4 мм",
    ]

    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    x = np.arange(3)
    dose_fluence = np.array(
        [
            float(row["Dmean_times_field_area_Gy_mm2_per_neutron"])
            for row in order
        ]
    )
    axes[0].bar(x, dose_fluence / dose_fluence[0], color="#4c78a8")
    axes[0].axhline(1.0, color="0.35", ls="--", lw=1.0)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Относительная доза на единицу флюенса")
    axes[0].set_title("Чувствительность дозового коэффициента")
    axes[0].grid(axis="y", alpha=0.22)

    all_let = [float(row["LETd_w_all_keV_um"]) for row in order]
    h_let = [float(row["LETd_w_hydrogen_keV_um"]) for row in order]
    width = 0.36
    axes[1].bar(x - width / 2, all_let, width, color="#6f3c9f", label="все")
    axes[1].bar(x + width / 2, h_let, width, color="#198f70", label="H")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel(r"$LET_{D,w}$, кэВ/мкм")
    axes[1].set_title("Чувствительность LET")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend()

    fig.savefig(args.output / "neutron_ng14_source_sensitivity.png", dpi=220)
    fig.savefig(args.output / "neutron_ng14_source_sensitivity.pdf")
    plt.close(fig)
    configurator.restore_original_styles()

    lines = [
        "# Чувствительность нейтронного расчёта к энергии и радиусу поля",
        "",
        (
            "Все три прогона выполнены на одинаковых 10 000 историях и "
            "одинаковых seed; менялся только один параметр источника."
        ),
        "",
        "| Сценарий | Доза/флюенс, изменение | LET(all), кэВ/мкм | LET(H), кэВ/мкм | H-доза, % |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, row in zip(labels, order):
        lines.append(
            "| "
            + label.replace("\n", ", ")
            + f" | {float(row['Dmean_times_field_area_Gy_mm2_per_neutron_relative_to_baseline_percent']):+.2f}%"
            + f" | {float(row['LETd_w_all_keV_um']):.3f}"
            + f" | {float(row['LETd_w_hydrogen_keV_um']):.3f}"
            + f" | {float(row['hydrogen_dose_fraction_percent']):.2f} |"
        )
    lines.extend(
        [
            "",
            (
                "Сравнение радиусов проводят по дозе на единицу флюенса, "
                "поскольку доза на одну первичную частицу неизбежно зависит "
                "от площади равномерного исходного диска."
            ),
            "",
            (
                "Эти короткие прогоны оценивают только устойчивость направления "
                "эффекта; полная LET тяжёлых фрагментов проверяется по основной "
                "серии из трёх независимых seed."
            ),
        ]
    )
    (args.output / "NEUTRON_NG14_SOURCE_SENSITIVITY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Written {csv_path}")


if __name__ == "__main__":
    main()
