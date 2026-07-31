"""Plot the article-based C-12 through-field reconstruction."""

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


ROOT = Path(
    r"C:\dev\dissertation\task4_5"
    r"\scoring_v2_carbon_c12_rat_through_physics"
)
RUNS_225 = (
    "opt4_1000_434MeVu_water225mm",
    "opt4_1000_434MeVu_water225mm_seed51031_77041",
    "opt4_1000_434MeVu_water225mm_seed88069_33091",
)
OUTPUT_STEM = ROOT / "carbon_c12_article_through_reconstruction"

# Digitised from Figure 4 of kizilova2025.pdf. These are deliberately
# reported as approximate values rather than as the original source data.
ARTICLE_CENTRE = 14.0
ARTICLE_LOW = 12.8
ARTICLE_HIGH = 15.1


def summary(run: str) -> dict[str, str]:
    path = ROOT / run / "gtv_analysis" / "gtv_summary.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.DictReader(handle))


def value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    no_water = value(
        summary("opt4_1000"),
        "gtv_LETd_w_keV_um",
    )
    water_200 = value(
        summary("opt4_1000_434MeVu_water200mm"),
        "gtv_LETd_w_keV_um",
    )
    seed_values = np.asarray(
        [
            value(summary(run), "gtv_LETd_w_keV_um")
            for run in RUNS_225
        ]
    )
    seed_mean = float(np.mean(seed_values))
    seed_sd = float(np.std(seed_values, ddof=1))

    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        plt.rcParams.update(
            {
                "font.size": 10.5,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "legend.fontsize": 9,
            }
        )
        figure, (ax_depth, ax_reference) = plt.subplots(
            1,
            2,
            figsize=(11.2, 4.9),
            constrained_layout=True,
        )

        ax_depth.axhspan(
            ARTICLE_LOW,
            ARTICLE_HIGH,
            color="#90caf9",
            alpha=0.28,
            label="рис. 4 статьи: ≈12,8–15,1",
        )
        depths = np.asarray([0.0, 200.0, 225.0])
        lets = np.asarray([no_water, water_200, seed_mean])
        ax_depth.plot(
            depths,
            lets,
            color="#6a1b9a",
            marker="o",
            linewidth=2.0,
            markersize=6,
            label="QGSP_INCLXX + Opt4",
        )
        ax_depth.errorbar(
            [225.0],
            [seed_mean],
            yerr=[seed_sd],
            fmt="none",
            ecolor="#6a1b9a",
            elinewidth=1.5,
            capsize=4,
        )
        for depth, let_value in zip(depths, lets):
            ax_depth.annotate(
                f"{let_value:.2f}",
                (depth, let_value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax_depth.set(
            xlabel="Водоэквивалентный путь до КТ-фантома, мм",
            ylabel=r"$LET_D$ GTVp, кэВ/мкм",
            title="Предмишенный водный путь устраняет расхождение",
            xlim=(-10.0, 240.0),
            ylim=(9.0, 16.3),
        )
        ax_depth.grid(alpha=0.2)
        ax_depth.legend(loc="lower right", frameon=True)
        ax_depth.tick_params(axis="both", labelsize=9.5)

        article_xerr = np.asarray(
            [[ARTICLE_CENTRE - ARTICLE_LOW], [ARTICLE_HIGH - ARTICLE_CENTRE]]
        )
        ax_reference.errorbar(
            ARTICLE_CENTRE,
            1.0,
            xerr=article_xerr,
            fmt="D",
            color="#263238",
            capsize=5,
            markersize=7,
            label="статья, рис. 4 (оцифровка)",
        )
        jitter = np.asarray([-0.08, 0.0, 0.08])
        ax_reference.scatter(
            seed_values,
            0.5 + jitter,
            color="#ef6c00",
            s=42,
            zorder=3,
            label="три seed, по 1000 ионов",
        )
        ax_reference.errorbar(
            seed_mean,
            0.5,
            xerr=seed_sd,
            fmt="s",
            color="#ad1457",
            capsize=5,
            markersize=8,
            label=f"среднее ± SD: {seed_mean:.2f} ± {seed_sd:.2f}",
        )
        ax_reference.axvspan(
            ARTICLE_LOW,
            ARTICLE_HIGH,
            color="#90caf9",
            alpha=0.20,
        )
        ax_reference.set_yticks(
            [0.5, 1.0],
            ["текущий пилот", "опубликованный референс"],
        )
        ax_reference.set(
            xlabel=r"$LET_D$, кэВ/мкм",
            title="Согласование с опубликованным прострелом",
            xlim=(9.0, 17.0),
            ylim=(0.15, 1.3),
        )
        ax_reference.grid(axis="x", alpha=0.2)
        ax_reference.legend(loc="upper left", frameon=True)
        ax_reference.tick_params(axis="both", labelsize=9.5)

        figure.suptitle(
            (
                r"$^{12}$C, прострел: воспроизведение постановки статьи; "
                "434 МэВ/нуклон; Opt4\n"
                "225 мм — водоэквивалентная реконструкция, "
                "не независимая валидация координаты кессона"
            ),
            fontsize=13,
        )
        for suffix in (".png", ".svg", ".pdf"):
            figure.savefig(OUTPUT_STEM.with_suffix(suffix), dpi=260)
        plt.close(figure)
    finally:
        configurator.restore_original_styles()

    print(f"output={OUTPUT_STEM}")
    print(f"article_approx={ARTICLE_CENTRE} [{ARTICLE_LOW}, {ARTICLE_HIGH}]")
    print(f"seed_values={seed_values.tolist()}")
    print(f"mean={seed_mean}")
    print(f"sd={seed_sd}")
    print(f"cv_percent={100.0 * seed_sd / seed_mean}")


if __name__ == "__main__":
    main()
