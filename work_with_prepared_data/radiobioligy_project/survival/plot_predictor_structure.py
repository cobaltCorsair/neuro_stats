"""What the predictors are made of, and why the dose slope keeps inverting.

A plain correlation matrix of the fitted design would be 97 columns of mostly
arithmetic: the quadratic dose term correlates with dose because it is dose
squared. The informative object is the *series-level* predictor set the model is
actually built from, and — more to the point — how much of each predictor is
already accounted for by the radiation family and the calendar year.

That second panel is the one that explains the session's recurring finding. Where
a predictor is largely determined by family, its marginal coefficient is not a
dose effect; it is a family effect wearing a dose label, which is why the marginal
and within-family slopes disagree in sign.

Two panels, two jobs, two colour treatments. Correlation is signed, so it takes a
diverging ramp with a neutral midpoint at zero. The variance share is unsigned and
bounded at zero and one, so it takes a single sequential hue, light to dark.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


BASE_SCRIPT = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель"
    r"\4.6_Ковариаты_pole_gamma\run_growth_prediction.py"
)
DEFAULT_OUTPUT = Path(r"C:\dev\dissertation_text\figures\figure_predictor_structure")

# Diverging pair and sequential hue taken unchanged from the reference palette:
# blue and red poles about a neutral grey, and the blue ramp light to dark.
DIVERGING = LinearSegmentedColormap.from_list(
    "corr", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a09f", "#e34948", "#8f2020"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "share", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
INK = "#0b0b0b"
INK_MUTED = "#52514e"

CONTINUOUS = {
    "total_dose_gy": "суммарная доза",
    "sum_d2_gy2": "сумма квадратов доз",
    "n_events": "число воздействий",
    "duration_hours": "длительность курса",
    "mean_interval_hours": "средний интервал",
    "log_v0": "исходный объём, log",
    "is_mixed": "смешанный режим",
}
CATEGORICAL = {
    "family": "семейство излучения",
    "year": "календарный год",
    "regimen_class": "класс режима",
    "order": "порядок компонентов",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def variance_share(values: np.ndarray, labels: np.ndarray) -> float:
    """Share of a predictor's variance lying between the groups of a factor.

    Bias-corrected, so a factor with many thin levels -- the calendar year has
    twelve -- cannot score high merely by having them. A negative correction is
    reported as zero: it means the factor explains less than chance would.
    """
    groups = [values[labels == level] for level in np.unique(labels)]
    groups = [g for g in groups if len(g) > 0]
    grand = values.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(((values - grand) ** 2).sum())
    k, n = len(groups), len(values)
    if ss_total <= 0 or n <= k:
        return float("nan")
    ms_within = (ss_total - ss_between) / (n - k)
    omega = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    return max(float(omega), 0.0)


def annotate(axis, matrix: np.ndarray, fmt: str, threshold: float) -> None:
    """Values on every cell -- in a matrix the numbers are the content.

    Ink colour flips on dark cells for contrast; it never takes the ramp colour.
    """
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            axis.text(
                j, i, format(value, fmt).replace("0.", ".").replace("-", "\u2212"),
                ha="center", va="center", fontsize=8,
                color="#ffffff" if abs(value) > threshold else INK,
            )


def figure(series: pd.DataFrame, output: Path) -> None:
    names = list(CONTINUOUS)
    labels = [CONTINUOUS[n] for n in names]
    block = series[names].astype(float)
    correlation = np.array(block.corr(method="pearson").to_numpy(), dtype=float)
    # Upper triangle carries no information a symmetric matrix has not already
    # shown, and the diagonal is ones by construction; blanking both halves the
    # reading load and stops the eye being pulled to cells that say nothing.
    correlation[np.triu_indices_from(correlation, k=0)] = np.nan

    shares = np.array([
        [variance_share(block[n].to_numpy(float), series[c].astype(str).to_numpy())
         for c in CATEGORICAL]
        for n in names
    ])

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13.4, 6.0), gridspec_kw={"width_ratios": (1.35, 1.0)}
    )
    fig.patch.set_facecolor("#fcfcfb")

    im = left.imshow(correlation, cmap=DIVERGING, vmin=-1, vmax=1)
    left.set_xticks(range(len(names)), labels, rotation=38, ha="right", fontsize=9, color=INK)
    left.set_yticks(range(len(names)), labels, fontsize=9, color=INK)
    left.set_title("Попарная корреляция предикторов\nкоэффициент Пирсона по 104 облучённым сериям",
                   fontsize=10.5, color=INK, pad=12)
    annotate(left, correlation, ".2f", 0.62)
    bar = fig.colorbar(im, ax=left, fraction=0.043, pad=0.03, ticks=[-1, -0.5, 0, 0.5, 1])
    bar.ax.tick_params(labelsize=8, colors=INK_MUTED)
    bar.outline.set_visible(False)

    im2 = right.imshow(shares, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    right.set_xticks(range(len(CATEGORICAL)), list(CATEGORICAL.values()),
                     rotation=38, ha="right", fontsize=9, color=INK)
    right.set_yticks(range(len(names)), labels, fontsize=9, color=INK)
    right.set_title("Какая доля предиктора уже задана группировкой\nисправленная на смещение доля дисперсии",
                    fontsize=10.5, color=INK, pad=12)
    annotate(right, shares, ".2f", 0.55)
    bar2 = fig.colorbar(im2, ax=right, fraction=0.055, pad=0.03, ticks=[0, 0.25, 0.5, 0.75, 1])
    bar2.ax.tick_params(labelsize=8, colors=INK_MUTED)
    bar2.outline.set_visible(False)

    for axis in (left, right):
        axis.set_xticks(np.arange(-0.5, axis.get_xlim()[1] + 0.5, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
        # A 2px surface gap between cells, as between any adjacent fills.
        axis.grid(which="minor", color="#fcfcfb", linewidth=2.0)
        axis.tick_params(which="minor", length=0)
        for side in axis.spines.values():
            side.set_visible(False)

    fig.suptitle(
        "Строение набора предикторов: чем они связаны друг с другом и что из них "
        "уже определено семейством и годом",
        fontsize=11.5, color=INK,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    print("=== доля дисперсии предиктора, объяснённая группировкой ===")
    table = pd.DataFrame(shares, index=labels, columns=list(CATEGORICAL.values()))
    print(table.round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    base = load_module("predictor_base", BASE_SCRIPT)
    _, evaluation, _, _ = base.build_longitudinal_tables()
    data = base.collapse_animal_daily_to_series(evaluation)
    series = (
        data.loc[data["is_treated"].astype(int) == 1]
        .drop_duplicates("series_key")
        .reset_index(drop=True)
    )
    print(f"серий: {len(series)}")
    figure(series, args.output)
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
