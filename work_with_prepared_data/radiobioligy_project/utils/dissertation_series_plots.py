"""Publication-oriented plots for retrospective calendar-series analyses.

The functions in this module belong to the radiobiology plotting layer of
``neuro_stats``.  They intentionally accept table-like dictionaries so audited
analysis protocols can reuse the application's visual style without importing
the Qt interface or rebuilding source Excel readers.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (
    MatplotlibConfigurator,
)


MATCHED_CONTROL_KINDS = {"same_date", "explicit_date_map"}


def _as_float(row: Mapping[str, object], key: str) -> float:
    return float(row[key])


def _series_marker(control_kind: str) -> dict[str, object]:
    """Return an outcome-independent marker style for a control source."""
    if control_kind in MATCHED_CONTROL_KINDS:
        return {
            "marker": "o",
            "facecolors": "#2b8cbe",
            "edgecolors": "#0868ac",
            "linewidths": 0.8,
        }
    if control_kind == "pooled_historical_fallback":
        return {
            "marker": "o",
            "facecolors": "none",
            "edgecolors": "#636363",
            "linewidths": 1.2,
        }
    return {
        "marker": "D",
        "facecolors": "#fdae6b",
        "edgecolors": "#e6550d",
        "linewidths": 0.9,
    }


def plot_series_dose_response(
    series_rows: Sequence[Mapping[str, object]],
    dose_rows: Sequence[Mapping[str, object]],
    *,
    endpoint: str,
    output_path: str | Path,
    family_labels: Mapping[str, str],
    families: Sequence[str],
    y_label: str,
    title: str,
    seed: int = 0,
    formats: Sequence[str] = ("png", "svg", "pdf"),
    dpi: int = 300,
) -> list[Path]:
    """Plot independent calendar-series estimates and hierarchical intervals.

    A line is deliberately not drawn between dose-group means: the audited
    protocol compares observed dose groups and does not fit a continuous dose
    response.  Marker fill communicates the control source, while red diamonds
    and error bars show the group mean and hierarchical 95% interval.
    """
    selected_families = list(families)
    if not selected_families:
        raise ValueError("At least one radiation family is required")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        ncols = 3
        nrows = math.ceil(len(selected_families) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(17.5, 5.4 * nrows),
            constrained_layout=True,
            squeeze=False,
        )
        rng = np.random.default_rng(seed)

        for axis, family in zip(axes.ravel(), selected_families):
            points = [
                row
                for row in series_rows
                if str(row["family"]) == family
                and str(row["regimen_class"]) == "single_fraction"
            ]
            summaries = [
                row
                for row in dose_rows
                if str(row["family"]) == family
                and str(row["regimen_class"]) == "single_fraction"
            ]

            for row in points:
                dose = _as_float(row, "dose_group_physical_gy")
                x = dose + float(rng.uniform(-0.10, 0.10))
                axis.scatter(
                    x,
                    _as_float(row, endpoint),
                    s=48,
                    alpha=0.88,
                    zorder=2,
                    **_series_marker(str(row.get("control_kind", "unknown"))),
                )

            for row in sorted(summaries, key=lambda item: _as_float(item, "dose_group_physical_gy")):
                dose = _as_float(row, "dose_group_physical_gy")
                mean = _as_float(row, f"{endpoint}_mean")
                low = _as_float(row, f"{endpoint}_hierarchical_p2_5")
                high = _as_float(row, f"{endpoint}_hierarchical_p97_5")
                axis.errorbar(
                    dose,
                    mean,
                    yerr=[[mean - low], [high - mean]],
                    fmt="D",
                    color="#cb181d",
                    markerfacecolor="#fb6a4a",
                    markeredgecolor="#99000d",
                    markersize=7,
                    linewidth=1.8,
                    capsize=4,
                    zorder=4,
                )
                axis.annotate(
                    f"n={int(float(row['n_series']))}",
                    (dose, high),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    color="#67000d",
                    annotation_clip=True,
                    clip_on=True,
                )

            axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
            axis.set_title(family_labels.get(family, family))
            axis.set_xlabel("Физическая доза, Гр")
            axis.set_ylabel(y_label)
            axis.grid(alpha=0.22)

            point_values = [_as_float(row, endpoint) for row in points]
            interval_highs = [
                _as_float(row, f"{endpoint}_hierarchical_p97_5") for row in summaries
            ]
            interval_lows = [
                _as_float(row, f"{endpoint}_hierarchical_p2_5") for row in summaries
            ]
            finite_values = [
                value
                for value in point_values + interval_highs + interval_lows + [0.0]
                if np.isfinite(value)
            ]
            if finite_values:
                data_bottom = min(finite_values)
                data_top = max(finite_values)
                data_span = max(data_top - min(0.0, data_bottom), 1.0)
                axis.set_ylim(
                    bottom=min(0.0, data_bottom - 0.03 * data_span),
                    top=data_top + 0.14 * data_span,
                )

            unique_doses = sorted({_as_float(row, "dose_group_physical_gy") for row in points})
            if len(unique_doses) == 1:
                axis.set_xlim(unique_doses[0] - 1.0, unique_doses[0] + 1.0)

        for axis in axes.ravel()[len(selected_families):]:
            axis.axis("off")

        legend_handles = [
            Line2D(
                [], [], marker="o", linestyle="none", markersize=8,
                markerfacecolor="#2b8cbe", markeredgecolor="#0868ac",
                label="одновременный или явно сопоставленный контроль",
            ),
            Line2D(
                [], [], marker="o", linestyle="none", markersize=8,
                markerfacecolor="none", markeredgecolor="#636363",
                label="объединённый исторический контроль",
            ),
            Line2D(
                [], [], marker="D", linestyle="none", markersize=8,
                markerfacecolor="#fb6a4a", markeredgecolor="#99000d",
                label="среднее по сериям и 95% bootstrap CI",
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="outside lower center",
            ncol=3,
            frameon=False,
            fontsize=13,
        )
        fig.suptitle(title, fontsize=24)

        created: list[Path] = []
        for extension in formats:
            target = output_path.with_suffix(f".{extension.lower()}")
            save_kwargs: dict[str, object] = {"bbox_inches": "tight"}
            if extension.lower() == "png":
                save_kwargs["dpi"] = dpi
            fig.savefig(target, **save_kwargs)
            created.append(target)
        plt.close(fig)
        return created
    finally:
        configurator.restore_original_styles()


def plot_calendar_block_forest(
    comparison_rows: Sequence[Mapping[str, object]],
    *,
    endpoint: str,
    output_path: str | Path,
    family_labels: Mapping[str, str],
    title: str,
    formats: Sequence[str] = ("png", "svg", "pdf"),
    dpi: int = 300,
) -> list[Path]:
    """Plot observed mean differences and calendar-block bootstrap intervals."""
    rows = [row for row in comparison_rows if str(row["endpoint"]) == endpoint]
    if not rows:
        raise ValueError(f"No calendar-block comparisons for endpoint {endpoint}")

    def comparison_label(row: Mapping[str, object]) -> str:
        family1 = str(row["family1"])
        family2 = str(row["family2"])
        dose1 = float(row["dose1_physical_gy"])
        dose2 = float(row["dose2_physical_gy"])
        if family1 == family2:
            return f"{family_labels.get(family1, family1)}: {dose1:g} − {dose2:g} Гр"
        return (
            f"{family_labels.get(family1, family1)} {dose1:g} − "
            f"{family_labels.get(family2, family2)} {dose2:g} Гр"
        )

    rows = sorted(rows, key=lambda row: float(row["observed_mean_difference"]))
    labels = [comparison_label(row) for row in rows]
    estimates = np.asarray([float(row["observed_mean_difference"]) for row in rows])
    lows = np.asarray([float(row["block_bootstrap_p2_5"]) for row in rows])
    highs = np.asarray([float(row["block_bootstrap_p97_5"]) for row in rows])
    y_positions = np.arange(len(rows), dtype=float)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        fig, axis = plt.subplots(
            figsize=(14.5, max(7.0, 0.55 * len(rows) + 2.4)),
            constrained_layout=True,
        )
        for y, row, estimate, low, high in zip(y_positions, rows, estimates, lows, highs):
            robust = bool(int(row["block_ci_excludes_zero"]))
            color = "#cb181d" if robust else "#737373"
            axis.errorbar(
                estimate,
                y,
                xerr=[[estimate - low], [high - estimate]],
                fmt="D",
                color=color,
                markerfacecolor="#fb6a4a" if robust else "#bdbdbd",
                markeredgecolor=color,
                markersize=7,
                linewidth=1.8,
                capsize=4,
                zorder=3,
            )
        axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
        axis.set_yticks(y_positions, labels)
        axis.set_xlabel("Разность средних (первая группа − вторая), процентные пункты")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
        axis.set_ylim(-0.8, len(rows) - 0.2)
        axis.legend(
            handles=[
                Line2D(
                    [], [], marker="D", linestyle="-", color="#cb181d",
                    markerfacecolor="#fb6a4a", label="95% блочный CI не включает 0",
                ),
                Line2D(
                    [], [], marker="D", linestyle="-", color="#737373",
                    markerfacecolor="#bdbdbd", label="95% блочный CI включает 0",
                ),
            ],
            loc="best",
            frameon=False,
            fontsize=13,
        )
        created: list[Path] = []
        for extension in formats:
            target = output_path.with_suffix(f".{extension.lower()}")
            save_kwargs: dict[str, object] = {"bbox_inches": "tight"}
            if extension.lower() == "png":
                save_kwargs["dpi"] = dpi
            fig.savefig(target, **save_kwargs)
            created.append(target)
        plt.close(fig)
        return created
    finally:
        configurator.restore_original_styles()


def plot_longitudinal_series_response(
    series_rows: Sequence[Mapping[str, object]],
    summary_rows: Sequence[Mapping[str, object]],
    availability_rows: Sequence[Mapping[str, object]],
    *,
    family: str,
    doses: Sequence[float],
    output_path: str | Path,
    family_label: str,
    title: str | None = None,
    formats: Sequence[str] = ("png", "svg", "pdf"),
    dpi: int = 300,
) -> list[Path]:
    """Plot calendar-series response, availability and recorded deaths."""
    selected_doses = [float(dose) for dose in doses]
    if not selected_doses:
        raise ValueError("At least one dose is required")
    selected_series = [
        row
        for row in series_rows
        if str(row["family"]) == family
        and str(row["regimen_class"]) == "single_fraction"
        and float(row["dose_group_physical_gy"]) in selected_doses
    ]
    selected_summary = [
        row
        for row in summary_rows
        if str(row["family"]) == family
        and str(row["regimen_class"]) == "single_fraction"
        and float(row["dose_group_physical_gy"]) in selected_doses
    ]
    selected_availability = [
        row
        for row in availability_rows
        if str(row["family"]) == family
        and str(row["regimen_class"]) == "single_fraction"
        and float(row["dose_group_physical_gy"]) in selected_doses
    ]
    if not selected_series or not selected_summary or not selected_availability:
        raise ValueError(f"Incomplete longitudinal inputs for {family}")

    colors = ["#2166ac", "#b2182b", "#4d9221", "#762a83"]
    dose_colors = {
        dose: colors[index % len(colors)] for index, dose in enumerate(selected_doses)
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(13.5, 11.5),
            sharex=True,
            gridspec_kw={"height_ratios": [4.2, 1.35, 1.1]},
            constrained_layout=True,
        )
        response_axis, availability_axis, death_axis = axes
        dose_handles: list[Line2D] = []

        for dose in selected_doses:
            color = dose_colors[dose]
            dose_series = [
                row
                for row in selected_series
                if float(row["dose_group_physical_gy"]) == dose
            ]
            series_keys = sorted({str(row["series_key"]) for row in dose_series})
            for series_key in series_keys:
                curve = sorted(
                    [row for row in dose_series if str(row["series_key"]) == series_key],
                    key=lambda row: float(row["day"]),
                )
                control_kind = str(curve[0].get("control_kind", "unknown"))
                response_axis.plot(
                    [float(row["day"]) for row in curve],
                    [float(row["volume_response"]) for row in curve],
                    color=color,
                    linewidth=1.0,
                    alpha=0.24,
                    linestyle="-" if control_kind in MATCHED_CONTROL_KINDS else ":",
                    zorder=1,
                )

            summary = sorted(
                [
                    row
                    for row in selected_summary
                    if float(row["dose_group_physical_gy"]) == dose
                ],
                key=lambda row: float(row["day"]),
            )
            days = np.asarray([float(row["day"]) for row in summary])
            means = np.asarray([float(row["mean_volume_response"]) for row in summary])
            lows = np.asarray([float(row["bootstrap_p2_5"]) for row in summary])
            highs = np.asarray([float(row["bootstrap_p97_5"]) for row in summary])
            response_axis.fill_between(days, lows, highs, color=color, alpha=0.18, zorder=2)
            response_axis.plot(days, means, color=color, linewidth=2.8, zorder=3)
            n_series = int(max(float(row["n_series"]) for row in summary))
            dose_handles.append(
                Line2D(
                    [], [], color=color, linewidth=2.8,
                    label=f"{dose:g} Гр: среднее и 95% CI (n={n_series})",
                )
            )

            availability = sorted(
                [
                    row
                    for row in selected_availability
                    if float(row["dose_group_physical_gy"]) == dose
                ],
                key=lambda row: float(row["day"]),
            )
            availability_axis.step(
                [float(row["day"]) for row in availability],
                [float(row["n_available_animals"]) for row in availability],
                where="post",
                color=color,
                linewidth=2.0,
                label=f"{dose:g} Гр",
            )
            death_axis.step(
                [float(row["day"]) for row in availability],
                [float(row["n_registered_deaths_cumulative"]) for row in availability],
                where="post",
                color=color,
                linewidth=2.0,
            )

        response_axis.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        response_axis.set_ylabel(r"Относительный объём $R_{exp}/R_{ctrl}$")
        response_axis.grid(alpha=0.22)
        response_axis.set_ylim(bottom=0.0)
        response_axis.legend(
            handles=dose_handles
            + [
                Line2D([], [], color="#555555", linewidth=1.2, label="серии: сопоставленный контроль"),
                Line2D([], [], color="#555555", linewidth=1.2, linestyle=":", label="серии: исторический контроль"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=2,
            frameon=False,
            fontsize=11,
        )

        availability_axis.set_ylabel("Доступно\nживотных")
        availability_axis.grid(alpha=0.22)
        availability_axis.set_ylim(bottom=0.0)
        availability_axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

        death_axis.set_ylabel("Смертей\nзарегистрировано")
        death_axis.set_xlabel("Сутки после облучения")
        death_axis.grid(alpha=0.22)
        maximum_deaths = max(
            float(row["n_registered_deaths_cumulative"])
            for row in selected_availability
        )
        if maximum_deaths == 0.0:
            death_axis.set_ylim(-0.05, 1.0)
            death_axis.set_yticks([0])
        else:
            death_axis.set_ylim(-0.05, maximum_deaths + max(0.25, 0.08 * maximum_deaths))
            death_axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        death_axis.set_xlim(0.0, 21.0)
        death_axis.set_xticks([0, 3, 7, 10, 14, 17, 21])

        fig.suptitle(
            title or f"{family_label}: продольный опухолевый ответ",
            fontsize=22,
        )
        created: list[Path] = []
        for extension in formats:
            target = output_path.with_suffix(f".{extension.lower()}")
            save_kwargs: dict[str, object] = {"bbox_inches": "tight"}
            if extension.lower() == "png":
                save_kwargs["dpi"] = dpi
            fig.savefig(target, **save_kwargs)
            created.append(target)
        plt.close(fig)
        return created
    finally:
        configurator.restore_original_styles()
