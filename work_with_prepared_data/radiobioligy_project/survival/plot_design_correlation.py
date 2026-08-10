"""The full 97-by-97 correlation of the fitted design, block by block.

Nine thousand four hundred and nine cells cannot be labelled or annotated, so the
only way this reads is by ordering. The features are grouped into the blocks the
model is actually built from -- the growth basis, the dose and regimen terms, the
configuration covariates, then seventy family interactions, and finally regimen
and order indicators -- and the blocks are separated by rules and named on the
margin. What the picture then shows is block structure, not individual pairs.

Correlations are taken over the 2596 series-day rows the ridge actually sees, not
over the 104 series: that is the matrix the penalty operates on. Constant columns
have no correlation to report and are drawn as gaps rather than as zeros.

The listing under the figure is the part to act on. A matrix this size is read for
its worst pairs, and printing them ranked saves hunting for dark cells by eye.
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
DEFAULT_OUTPUT = Path(r"C:\dev\dissertation_text\figures\figure_design_correlation")

DIVERGING = LinearSegmentedColormap.from_list(
    "corr", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a09f", "#e34948", "#8f2020"]
)
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"

GROWTH = (
    "time", "time2", "time3", "hinge_day3_sq", "hinge_day7_sq", "hinge_day14_sq",
    "log_v0_x_time", "log_v0_x_time2",
)
DOSE = (
    "treated_x_g12", "dose_x_g12", "sum_d2_x_g12", "sqrt_dose_x_g12", "dose2_x_g12",
    "n_events_x_g12", "duration_x_g12", "mean_interval_x_g12", "timing_known_x_g12",
    "mixed_x_g12",
)
CONFIGURATION = (
    "e_applicator_mismatch_dose_g12", "y_field_enlargement_g12", "y_field_known_g12",
    "e_applicator_known_g12",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def blocks(names: list[str]) -> list[tuple[str, list[int]]]:
    """Group the columns into the blocks the design is composed of.

    Family interactions are kept together and ordered by family, so a family's
    seven terms sit adjacent and its internal collinearity reads as one square
    rather than as scattered cells.
    """
    index = {name: i for i, name in enumerate(names)}
    ordered: list[tuple[str, list[int]]] = [
        ("базис роста", [index[n] for n in GROWTH if n in index]),
        ("доза и режим", [index[n] for n in DOSE if n in index]),
        ("конфигурация", [index[n] for n in CONFIGURATION if n in index]),
    ]
    families: dict[str, list[int]] = {}
    for name, i in index.items():
        if not name.startswith("family_"):
            continue
        family = name[len("family_"):].rsplit("_", 1)[0]
        # Longest suffix first: "_g7" would otherwise match "_dose_g7" and split
        # a family across two blocks under a name it does not have.
        suffixes = ("_dose_g12", "_sumd2_g12", "_dose_g21", "_dose_g7", "_g21", "_g12", "_g7")
        for suffix in sorted(suffixes, key=len, reverse=True):
            if name.endswith(suffix):
                family = name[len("family_"):-len(suffix)]
                break
        families.setdefault(family, []).append(i)
    for family in sorted(families):
        ordered.append((family, sorted(families[family])))
    tail = [index[n] for n in names if n.startswith(("regimen_", "order_"))]
    ordered.append(("режим и порядок", sorted(tail)))
    return [(label, cols) for label, cols in ordered if cols]


def figure(matrix: np.ndarray, groups: list[tuple[str, list[int]]], output: Path) -> None:
    order = [i for _, cols in groups for i in cols]
    shown = matrix[np.ix_(order, order)]
    edges, position = [], 0
    for _, cols in groups:
        position += len(cols)
        edges.append(position)

    fig, axis = plt.subplots(figsize=(12.6, 11.2))
    fig.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    image = axis.imshow(shown, cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")

    # Rules between blocks in the surface colour, the same 2px separation used
    # between any adjacent fills.
    for edge in edges[:-1]:
        axis.axhline(edge - 0.5, color=SURFACE, linewidth=2.0)
        axis.axvline(edge - 0.5, color=SURFACE, linewidth=2.0)

    centres = [(edges[i] + (edges[i - 1] if i else 0)) / 2 - 0.5 for i in range(len(groups))]
    names = [label for label, _ in groups]
    axis.set_xticks(centres, names, rotation=62, ha="right", fontsize=8.5, color=INK)
    axis.set_yticks(centres, names, fontsize=8.5, color=INK)
    axis.tick_params(length=0)
    for side in axis.spines.values():
        side.set_visible(False)

    bar = fig.colorbar(image, ax=axis, fraction=0.036, pad=0.02,
                       ticks=[-1, -0.5, 0, 0.5, 1])
    bar.ax.tick_params(labelsize=8, colors=INK_MUTED)
    bar.outline.set_visible(False)
    bar.set_label("коэффициент корреляции Пирсона", fontsize=9, color=INK_MUTED)

    axis.set_title(
        "Полная корреляционная матрица подобранной модели: 97 признаков, 2596 строк\n"
        "признаки упорядочены по блокам; подписи на полях именуют блок, а не отдельный столбец",
        fontsize=11, color=INK, pad=14,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    base = load_module("design_base", BASE_SCRIPT)
    _, evaluation, _, _ = base.build_longitudinal_tables()
    data = base.collapse_animal_daily_to_series(evaluation)
    categories = base.fixed_categories(data)
    design, names = base.build_features(data, "hierarchical_robust", categories)

    frame = pd.DataFrame(np.asarray(design, dtype=float), columns=names)
    constant = [n for n in names if frame[n].std(ddof=0) == 0]
    matrix = np.array(frame.corr(method="pearson").to_numpy(), dtype=float)

    groups = blocks(names)
    figure(matrix, groups, args.output)

    upper = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    finite = upper & np.isfinite(matrix)
    values = np.abs(matrix[finite])
    pairs = [
        (abs(matrix[i, j]), matrix[i, j], names[i], names[j])
        for i, j in zip(*np.where(finite)) if abs(matrix[i, j]) >= 0.90
    ]
    pairs.sort(reverse=True)

    print(f"признаков {len(names)}, строк {design.shape[0]}, "
          f"постоянных столбцов {len(constant)}")
    print(f"пар с определённой корреляцией: {int(finite.sum())}")
    for threshold in (0.7, 0.8, 0.9, 0.95, 0.99):
        print(f"  |r| >= {threshold:.2f}: {int((values >= threshold).sum())}")
    print(f"\n=== пары с |r| >= 0.90 ({len(pairs)}) ===")
    for _, value, left, right in pairs[:25]:
        print(f"  {value:+.3f}  {left}  <->  {right}")
    if len(pairs) > 25:
        print(f"  ... ещё {len(pairs) - 25}")
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
