"""Summarise 10k confirmation runs selected by the geometry screen."""

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
)
from work_with_prepared_data.radiobioligy_project.survival.analyze_geant4_proton_100MeV_reduced_gtv import (  # noqa: E402
    DEFAULT_CT_DIR,
    DEFAULT_RS,
    gtv_geometry,
)
from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (  # noqa: E402
    MatplotlibConfigurator,
)


SEED_PAIRS = ((50131, 80309), (60131, 90309))


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
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=Path(
            r"C:\dev\dissertation\task4_5"
            r"\scoring_v2_electron_novac_rat"
            r"\analysis_geometry_screen"
            r"\confirmation_selection.json"
        ),
    )
    parser.add_argument("--histories", type=int, default=10000)
    parser.add_argument("--energy-mev", type=float, default=10.0)
    parser.add_argument("--ct-dir", type=Path, default=DEFAULT_CT_DIR)
    parser.add_argument("--rtstruct", type=Path, default=DEFAULT_RS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(
        args.selection_json.read_text(encoding="utf-8")
    )
    _, gtv_ids, gtv_lookup, gtv_depth_interval = gtv_geometry(
        args.ct_dir,
        args.rtstruct,
    )
    rows: list[dict[str, object]] = []
    metrics = (
        "D90_over_Dmean",
        "D95_over_Dmean",
        "D98_over_Dmean",
        "HI98",
        "LETd_w_GTVp_keV_um",
    )
    for replicate, (seed1_base, seed2_base) in enumerate(
        SEED_PAIRS,
        start=1,
    ):
        base_name = (
            f"livermore_{args.histories}_seed"
            f"{seed1_base}_{seed2_base}"
        )
        case_specs = [
            ("nominal", args.root_dir / base_name),
            (
                str(selection["worst_translation"]),
                args.root_dir
                / f"{base_name}_confirm_{selection['worst_translation']}",
            ),
            (
                str(selection["worst_tilt"]),
                args.root_dir
                / f"{base_name}_confirm_{selection['worst_tilt']}",
            ),
        ]
        replicate_rows: list[dict[str, object]] = []
        for label, root in case_specs:
            summary, _ = analyse_rat_case(
                root,
                args.energy_mev,
                gtv_ids,
                gtv_lookup,
                gtv_depth_interval,
            )
            replicate_rows.append(
                {
                    "replicate": replicate,
                    "seed1_base": seed1_base,
                    "seed2_base": seed2_base,
                    "scenario": label,
                    **summary,
                }
            )
        baseline = replicate_rows[0]
        for row in replicate_rows:
            for metric in metrics:
                row[f"delta_{metric}"] = (
                    float(row[metric]) - float(baseline[metric])
                )
                row[f"relative_{metric}_percent"] = 100.0 * (
                    float(row[metric]) / float(baseline[metric]) - 1.0
                )
        rows.extend(replicate_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (
        args.output_dir / "electron_geometry_confirmation.csv"
    ).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows: list[dict[str, object]] = []
    for scenario in (
        "nominal",
        str(selection["worst_translation"]),
        str(selection["worst_tilt"]),
    ):
        selected = [row for row in rows if row["scenario"] == scenario]
        aggregate: dict[str, object] = {
            "scenario": scenario,
            "n_replicates": len(selected),
        }
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in selected],
                dtype=float,
            )
            aggregate[f"{metric}_mean"] = float(np.mean(values))
            aggregate[f"{metric}_min"] = float(np.min(values))
            aggregate[f"{metric}_max"] = float(np.max(values))
            deltas = np.asarray(
                [float(row[f"delta_{metric}"]) for row in selected],
                dtype=float,
            )
            aggregate[f"delta_{metric}_mean"] = float(np.mean(deltas))
            aggregate[f"delta_{metric}_min"] = float(np.min(deltas))
            aggregate[f"delta_{metric}_max"] = float(np.max(deltas))
        aggregate_rows.append(aggregate)
    with (
        args.output_dir / "electron_geometry_confirmation_summary.csv"
    ).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(aggregate_rows[0]),
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 13,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "figure.titlesize": 14,
            }
        )
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(12.0, 4.5),
            constrained_layout=False,
        )
        figure.subplots_adjust(
            left=0.06,
            right=0.99,
            bottom=0.18,
            top=0.80,
            wspace=0.38,
        )
        scenario_labels = {
            str(selection["worst_translation"]): "смещение z −3 мм",
            str(selection["worst_tilt"]): "наклон z +5°",
        }
        colours = ("#2f6db2", "#d97706")
        for axis, metric, value_column, title, ylabel in (
            (
                axes[0],
                "D90_over_Dmean",
                "delta_D90_over_Dmean",
                r"$\Delta D_{90}/D_{\mathrm{mean}}$",
                "Абсолютное изменение",
            ),
            (
                axes[1],
                "D98_over_Dmean",
                "delta_D98_over_Dmean",
                r"$\Delta D_{98}/D_{\mathrm{mean}}$",
                "Абсолютное изменение",
            ),
            (
                axes[2],
                "LETd_w_GTVp_keV_um",
                "relative_LETd_w_GTVp_keV_um_percent",
                r"$\Delta LET_{D,w}$",
                "Относительное изменение, %",
            ),
        ):
            axis.axhline(0.0, color="#666666", lw=1.0)
            for index, scenario in enumerate(scenario_labels):
                selected = [
                    row
                    for row in rows
                    if row["scenario"] == scenario
                ]
                values = np.asarray(
                    [float(row[value_column]) for row in selected]
                )
                x = np.full(values.size, index, dtype=float)
                x += np.linspace(-0.06, 0.06, values.size)
                axis.scatter(
                    x,
                    values,
                    color=colours[index],
                    s=55,
                    zorder=3,
                    label=(
                        scenario_labels[scenario]
                        if metric == "D90_over_Dmean"
                        else None
                    ),
                )
                axis.plot(
                    [index - 0.18, index + 0.18],
                    [float(np.mean(values)), float(np.mean(values))],
                    color=colours[index],
                    lw=2.8,
                )
            axis.set(
                title=title,
                ylabel=ylabel,
                xticks=(0, 1),
                xticklabels=list(scenario_labels.values()),
            )
            axis.grid(axis="y", alpha=0.2)
        figure.suptitle(
            "Электроны 10 МэВ: геометрическая чувствительность "
            "(2 парные реализации × 10 000 частиц)",
            y=0.96,
        )
        for suffix in (".png", ".svg", ".pdf"):
            figure.savefig(
                args.output_dir
                / f"electron_geometry_confirmation_paired{suffix}",
                dpi=240 if suffix == ".png" else None,
                bbox_inches="tight",
            )
        plt.close(figure)
    finally:
        configurator.restore_original_styles()

    (args.output_dir / "confirmation_context.json").write_text(
        json.dumps(
            {
                "selection": selection,
                "confirmation_histories": args.histories,
                "confirmation_seed_pairs": [
                    list(pair) for pair in SEED_PAIRS
                ],
                "selection_is_screening_not_optimisation": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "electron_geometry_confirmation.csv")


if __name__ == "__main__":
    main()
