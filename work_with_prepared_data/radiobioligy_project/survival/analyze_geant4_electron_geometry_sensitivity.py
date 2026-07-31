"""Analyse paired geometric perturbations of the electron rat calculation.

The screen compares a nominal incident plane with pre-specified translations
and tilts.  It is a robustness analysis, not an optimisation of the beam
position against the resulting DVH.
"""

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

from work_with_prepared_data.radiobioligy_project.survival.analyze_geant4_electron_novac import (  # noqa: E402
    analyse_rat_case,
    normalised_dvh,
)
from work_with_prepared_data.radiobioligy_project.survival.analyze_geant4_proton_100MeV_reduced_gtv import (  # noqa: E402
    DEFAULT_CT_DIR,
    DEFAULT_RS,
    gtv_geometry,
)
from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (  # noqa: E402
    MatplotlibConfigurator,
)


SCENARIOS = (
    ("geom_base", "номинальная", "baseline"),
    ("shift_x_p3", "x +3 мм", "translation"),
    ("shift_x_m3", "x −3 мм", "translation"),
    ("shift_z_p3", "z +3 мм", "translation"),
    ("shift_z_m3", "z −3 мм", "translation"),
    ("tilt_x_p5", "наклон x +5°", "tilt"),
    ("tilt_x_m5", "наклон x −5°", "tilt"),
    ("tilt_z_p5", "наклон z +5°", "tilt"),
    ("tilt_z_m5", "наклон z −5°", "tilt"),
)

COLOURS = {
    "baseline": "#222222",
    "translation": "#2f6db2",
    "tilt": "#d97706",
}


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_root(
    base: Path,
    histories: int,
    seed1: int,
    seed2: int,
    tag: str,
) -> Path:
    return base / f"livermore_{histories}_seed{seed1}_{seed2}_{tag}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(
            r"C:\dev\dissertation\task4_5"
            r"\scoring_v2_electron_novac_rat"
        ),
    )
    parser.add_argument("--histories", type=int, default=2500)
    parser.add_argument("--seed1-base", type=int, default=55131)
    parser.add_argument("--seed2-base", type=int, default=85309)
    parser.add_argument("--energy-mev", type=float, default=10.0)
    parser.add_argument("--ct-dir", type=Path, default=DEFAULT_CT_DIR)
    parser.add_argument("--rtstruct", type=Path, default=DEFAULT_RS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, gtv_ids, gtv_lookup, gtv_depth_interval = gtv_geometry(
        args.ct_dir,
        args.rtstruct,
    )

    rows: list[dict[str, object]] = []
    profiles: dict[str, dict[str, np.ndarray]] = {}
    for tag, label, kind in SCENARIOS:
        root = run_root(
            args.root_dir,
            args.histories,
            args.seed1_base,
            args.seed2_base,
            tag,
        )
        summary, profile = analyse_rat_case(
            root,
            args.energy_mev,
            gtv_ids,
            gtv_lookup,
            gtv_depth_interval,
        )
        metadata = json.loads(
            (
                root
                / f"E{args.energy_mev:g}MeV"
                / "run_metadata.json"
            ).read_text(encoding="utf-8-sig")
        )
        row = {
            "scenario": tag,
            "display_name": label,
            "geometry_kind": kind,
            "source_x_mm": float(metadata.get("source_x_mm", 0.0)),
            "source_z_mm": float(metadata.get("source_z_mm", 0.0)),
            "tilt_x_deg": float(metadata.get("tilt_x_deg", 0.0)),
            "tilt_z_deg": float(metadata.get("tilt_z_deg", 0.0)),
            **summary,
        }
        rows.append(row)
        profiles[tag] = profile

    baseline = next(row for row in rows if row["scenario"] == "geom_base")
    for row in rows:
        for metric in (
            "D90_over_Dmean",
            "D98_over_Dmean",
            "HI98",
            "LETd_w_GTVp_keV_um",
        ):
            row[f"delta_{metric}"] = (
                float(row[metric]) - float(baseline[metric])
            )
        row["relative_LETd_change_percent"] = 100.0 * (
            float(row["LETd_w_GTVp_keV_um"])
            / float(baseline["LETd_w_GTVp_keV_um"])
            - 1.0
        )
    write_rows(args.output_dir / "electron_geometry_sensitivity.csv", rows)

    translation_rows = [
        row for row in rows if row["geometry_kind"] == "translation"
    ]
    tilt_rows = [row for row in rows if row["geometry_kind"] == "tilt"]
    worst_translation = min(
        translation_rows,
        key=lambda row: float(row["D90_over_Dmean"]),
    )
    worst_tilt = min(
        tilt_rows,
        key=lambda row: float(row["D90_over_Dmean"]),
    )
    selection = {
        "selection_rule": (
            "minimum screening D90/Dmean within each pre-specified "
            "perturbation class"
        ),
        "screen_histories": args.histories,
        "paired_seed_bases": [args.seed1_base, args.seed2_base],
        "worst_translation": worst_translation["scenario"],
        "worst_tilt": worst_tilt["scenario"],
        "confirmation_histories": 10000,
        "confirmation_seed_bases": [50131, 80309],
    }
    (args.output_dir / "confirmation_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 15,
                "axes.labelsize": 13,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
            }
        )
        figure, (axis_dvh, axis_delta, axis_let) = plt.subplots(
            3,
            1,
            figsize=(10.0, 12.5),
            constrained_layout=True,
            gridspec_kw={"height_ratios": (1.1, 1.0, 0.8)},
        )
        x_grid = np.linspace(0.0, 2.0, 501)
        for tag, label, kind in SCENARIOS:
            profile = profiles[tag]
            line_width = 2.8 if kind == "baseline" else 1.35
            alpha = 1.0 if kind == "baseline" else 0.62
            axis_dvh.plot(
                x_grid,
                normalised_dvh(profile["reporting_dose_Gy"], x_grid),
                color=COLOURS[kind],
                lw=line_width,
                alpha=alpha,
                label=label,
            )
        axis_dvh.set(
            title=(
                "Электроны 10 МэВ: геометрическая чувствительность DVH GTVp\n"
                f"парный скрининг, {args.histories:,} первичных частиц"
            ).replace(",", " "),
            xlabel=r"Доза в элементе / $D_{\mathrm{mean}}$ GTVp",
            ylabel="Объём GTVp, получивший ≥ дозы, %",
            xlim=(0.0, 1.8),
            ylim=(0.0, 101.0),
        )
        axis_dvh.grid(alpha=0.18)
        axis_dvh.legend(
            frameon=False,
            ncol=3,
            loc="lower left",
        )

        nonbaseline = [row for row in rows if row["geometry_kind"] != "baseline"]
        y_positions = np.arange(len(nonbaseline))
        delta_d90 = np.asarray(
            [float(row["delta_D90_over_Dmean"]) for row in nonbaseline]
        )
        delta_d98 = np.asarray(
            [float(row["delta_D98_over_Dmean"]) for row in nonbaseline]
        )
        axis_delta.axvline(0.0, color="#666666", lw=1.0)
        axis_delta.scatter(
            delta_d90,
            y_positions - 0.13,
            color="#2f6db2",
            marker="o",
            s=54,
            label=r"$\Delta D_{90}/D_{\mathrm{mean}}$",
            zorder=3,
        )
        axis_delta.scatter(
            delta_d98,
            y_positions + 0.13,
            color="#d97706",
            marker="s",
            s=48,
            label=r"$\Delta D_{98}/D_{\mathrm{mean}}$",
            zorder=3,
        )
        for y, d90, d98 in zip(y_positions, delta_d90, delta_d98):
            axis_delta.plot(
                [d90, d98],
                [y - 0.13, y + 0.13],
                color="#aaaaaa",
                lw=0.8,
                zorder=1,
            )
        axis_delta.set(
            title="Изменение нижней части дозового распределения относительно номинальной геометрии",
            xlabel="Абсолютное изменение нормированной дозы",
            yticks=y_positions,
            yticklabels=[str(row["display_name"]) for row in nonbaseline],
        )
        axis_delta.invert_yaxis()
        axis_delta.grid(axis="x", alpha=0.18)
        axis_delta.legend(frameon=False, ncol=2)

        let_delta = np.asarray(
            [float(row["relative_LETd_change_percent"]) for row in nonbaseline]
        )
        kinds = [str(row["geometry_kind"]) for row in nonbaseline]
        axis_let.axhline(0.0, color="#666666", lw=1.0)
        axis_let.bar(
            np.arange(len(nonbaseline)),
            let_delta,
            color=[COLOURS[kind] for kind in kinds],
            alpha=0.78,
        )
        axis_let.set(
            title=r"Устойчивость дозо-взвешенной $LET_{D,w}$ в GTVp",
            xlabel="Геометрический сценарий",
            ylabel="Изменение относительно номинального, %",
            xticks=np.arange(len(nonbaseline)),
            xticklabels=[
                str(row["display_name"]) for row in nonbaseline
            ],
        )
        axis_let.tick_params(axis="x", rotation=28)
        axis_let.grid(axis="y", alpha=0.18)

        for suffix in (".png", ".svg", ".pdf"):
            figure.savefig(
                args.output_dir
                / f"electron_geometry_sensitivity{suffix}",
                dpi=240 if suffix == ".png" else None,
                bbox_inches="tight",
            )
        plt.close(figure)
    finally:
        configurator.restore_original_styles()

    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "worst_translation": worst_translation["scenario"],
                "worst_tilt": worst_tilt["scenario"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
