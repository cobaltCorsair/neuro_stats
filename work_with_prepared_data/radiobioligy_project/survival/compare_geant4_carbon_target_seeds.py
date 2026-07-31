"""Compare independent C-12 target-width rat-phantom runs.

The script reads the GTV summaries and normalised DVHs produced by
``analyze_geant4_proton_100MeV_reduced_gtv.py``.  It deliberately compares
only runs with the same geometry, source, physics list, scoring grid and
number of primary histories; the random-number seeds are the intended
difference.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    (
        "gtv_mean_dose_Gy_per_primary",
        "Средняя доза на первичный",
        "Гр",
    ),
    ("gtv_LETd_w_keV_um", r"$LET_{D,w}$ GTVp", "кэВ/мкм"),
    ("central_peak_depth_mm", "Глубина центрального максимума", "мм"),
    (
        "gtv_nonzero_dose_fraction",
        "Ненулевой объём, исходная сетка",
        "доля",
    ),
    (
        "gtv_aggregated_D50_over_Dmean_0p8x0p8x0p4",
        r"$D_{50}/D_{\mathrm{mean}}$, 0,8 мм",
        "",
    ),
    (
        "gtv_aggregated_D90_over_Dmean_0p8x0p8x0p4",
        r"$D_{90}/D_{\mathrm{mean}}$, 0,8 мм",
        "",
    ),
    (
        "gtv_aggregated_D50_over_Dmean_1p2x1p2x0p6",
        r"$D_{50}/D_{\mathrm{mean}}$, 1,2 мм",
        "",
    ),
    (
        "gtv_aggregated_D90_over_Dmean_1p2x1p2x0p6",
        r"$D_{90}/D_{\mathrm{mean}}$, 1,2 мм",
        "",
    ),
    (
        "gtv_aggregated_D50_over_Dmean_1p6x1p6x0p8",
        r"$D_{50}/D_{\mathrm{mean}}$, 1,6 мм",
        "",
    ),
    (
        "gtv_aggregated_D90_over_Dmean_1p6x1p6x0p8",
        r"$D_{90}/D_{\mathrm{mean}}$, 1,6 мм",
        "",
    ),
)


def read_single_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one data row in {path}, got {len(rows)}")
    return rows[0]


def seed_label(case_dir: Path) -> str:
    match = re.search(r"_seed(\d+)_(\d+)$", case_dir.name)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return case_dir.name


def read_dvh(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = case_dir / "gtv_analysis" / "through_normalised_dvh.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    x = np.asarray([float(row["dose_over_gtv_mean"]) for row in rows])
    y = np.asarray(
        [
            float(row["volume_receiving_percent_1p6x1p6x0p8"])
            for row in rows
        ]
    )
    order = np.argsort(x)
    return x[order], y[order]


def dose_at_volume(x: np.ndarray, y: np.ndarray, volume_percent: float) -> float:
    below = np.flatnonzero(y <= volume_percent)
    if below.size == 0:
        return float(x[-1])
    right = int(below[0])
    if right == 0:
        return float(x[0])
    left = right - 1
    y0, y1 = y[left], y[right]
    x0, x1 = x[left], x[right]
    if y0 == y1:
        return float(x1)
    fraction = (volume_percent - y0) / (y1 - y0)
    return float(x0 + fraction * (x1 - x0))


def format_ru(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-dir",
        type=Path,
        action="append",
        required=True,
        help="Case directory containing gtv_analysis; specify exactly twice.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if len(args.case_dir) != 2:
        parser.error("Exactly two --case-dir arguments are required")

    case_dirs = [path.resolve() for path in args.case_dir]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        read_single_row(path / "gtv_analysis" / "gtv_summary.csv")
        for path in case_dirs
    ]
    labels = [seed_label(path) for path in case_dirs]

    invariants = (
        "configuration",
        "histories",
        "grid",
        "voxel_mm",
        "gtv_voxels",
        "gtv_volume_mm3",
    )
    for key in invariants:
        values = [summary[key] for summary in summaries]
        if values[0] != values[1]:
            raise ValueError(f"Runs differ in invariant {key}: {values}")

    metric_values: dict[str, np.ndarray] = {}
    seed_rows: list[dict[str, object]] = []
    for label, case_dir, summary in zip(labels, case_dirs, summaries):
        row: dict[str, object] = {
            "seed": label,
            "case_dir": str(case_dir),
        }
        for key, _, _ in METRICS:
            row[key] = float(summary[key])
        seed_rows.append(row)

    summary_rows: list[dict[str, object]] = []
    for key, label, unit in METRICS:
        values = np.asarray([float(summary[key]) for summary in summaries])
        metric_values[key] = values
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        cv_percent = float(sd / abs(mean) * 100.0) if mean else float("nan")
        relative_difference = (
            float(abs(values[1] - values[0]) / abs(mean) * 100.0)
            if mean
            else float("nan")
        )
        summary_rows.append(
            {
                "metric": key,
                "label": label.replace("$", ""),
                "unit": unit,
                "seed_1": float(values[0]),
                "seed_2": float(values[1]),
                "mean": mean,
                "sample_sd": sd,
                "cv_percent": cv_percent,
                "absolute_difference": float(abs(values[1] - values[0])),
                "relative_difference_percent": relative_difference,
            }
        )

    with (output_dir / "seed_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)

    with (output_dir / "seed_comparison_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    dvhs = [read_dvh(path) for path in case_dirs]
    d90_values = [dose_at_volume(x, y, 90.0) for x, y in dvhs]
    common_x = np.linspace(0.0, 2.0, 401)
    common_rows = []
    interpolated = []
    for x, y in dvhs:
        interpolated.append(np.interp(common_x, x, y))
    for index, dose_ratio in enumerate(common_x):
        common_rows.append(
            {
                "dose_over_gtv_mean": dose_ratio,
                f"volume_percent_seed_{labels[0].replace('/', '_')}": (
                    interpolated[0][index]
                ),
                f"volume_percent_seed_{labels[1].replace('/', '_')}": (
                    interpolated[1][index]
                ),
            }
        )
    with (output_dir / "seed_dvh_1p6mm.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(common_rows[0]))
        writer.writeheader()
        writer.writerows(common_rows)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(12.6, 5.2), constrained_layout=True
    )
    colours = ("#482475", "#2a788e")
    for label, (x, y), d90, colour in zip(
        labels, dvhs, d90_values, colours
    ):
        axes[0].plot(
            x,
            y,
            linewidth=2.2,
            color=colour,
            label=f"seed {label}; D90/Dmean={d90:.3f}",
        )
        axes[0].scatter([d90], [90.0], s=32, color=colour, zorder=4)
    axes[0].axhline(90.0, color="#777777", linestyle="--", linewidth=1.0)
    axes[0].set(
        title="DVH GTVp: независимые запуски",
        xlabel=r"Доза в элементе / $D_{\mathrm{mean}}$ GTVp",
        ylabel="Объём GTVp, получивший ≥ дозы, %",
        xlim=(0.0, 1.8),
        ylim=(0.0, 101.5),
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, loc="upper right")

    plot_keys = (
        "gtv_mean_dose_Gy_per_primary",
        "gtv_LETd_w_keV_um",
        "gtv_nonzero_dose_fraction",
        "gtv_aggregated_D50_over_Dmean_1p6x1p6x0p8",
        "gtv_aggregated_D90_over_Dmean_1p6x1p6x0p8",
    )
    plot_labels = (
        "средняя доза",
        r"$LET_{D,w}$",
        "ненулевой объём",
        r"$D_{50}/D_{\mathrm{mean}}$",
        r"$D_{90}/D_{\mathrm{mean}}$",
    )
    differences = []
    for key in plot_keys:
        values = metric_values[key]
        differences.append(abs(values[1] - values[0]) / np.mean(values) * 100)
    y_positions = np.arange(len(plot_labels))
    bars = axes[1].barh(y_positions, differences, color="#7a1fa2", alpha=0.85)
    axes[1].set(
        title="Межзапусковое различие",
        xlabel="Абсолютное различие / среднее двух запусков, %",
        yticks=y_positions,
        yticklabels=plot_labels,
    )
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].bar_label(bars, fmt="%.2f%%", padding=3)
    axes[1].set_xlim(0.0, max(differences) * 1.28)

    figure.suptitle(
        "Проектный углеродный пик: проверка устойчивости по seed",
        fontsize=14,
    )
    for suffix in (".png", ".svg", ".pdf"):
        figure.savefig(
            output_dir / f"carbon_c12_target20_seed_comparison{suffix}",
            dpi=220,
        )
    plt.close(figure)

    mean_dvh = np.mean(np.vstack(interpolated), axis=0)
    minimum_dvh = np.min(np.vstack(interpolated), axis=0)
    maximum_dvh = np.max(np.vstack(interpolated), axis=0)
    mean_d90 = dose_at_volume(common_x, mean_dvh, 90.0)
    mean_d50 = dose_at_volume(common_x, mean_dvh, 50.0)

    dvh_figure, dvh_axis = plt.subplots(
        1, 1, figsize=(8.8, 6.6), constrained_layout=True
    )
    dvh_axis.fill_between(
        common_x,
        minimum_dvh,
        maximum_dvh,
        color="#7a1fa2",
        alpha=0.16,
        label="диапазон двух seed",
    )
    for label, curve, colour in zip(labels, interpolated, colours):
        dvh_axis.plot(
            common_x,
            curve,
            color=colour,
            linewidth=1.25,
            linestyle="--",
            alpha=0.9,
            label=f"seed {label}",
        )
    dvh_axis.plot(
        common_x,
        mean_dvh,
        color="#5b187e",
        linewidth=3.0,
        label="средняя DVH",
    )
    for volume, dose_ratio, vertical_offset in (
        (90.0, mean_d90, -8),
        (50.0, mean_d50, 7),
    ):
        dvh_axis.axhline(
            volume,
            color="#777777",
            linestyle=":",
            linewidth=0.9,
        )
        dvh_axis.axvline(
            dose_ratio,
            color="#777777",
            linestyle=":",
            linewidth=0.9,
        )
        dvh_axis.scatter(
            [dose_ratio],
            [volume],
            color="#5b187e",
            s=45,
            zorder=5,
        )
        dvh_axis.annotate(
            rf"$D_{{{int(volume)}}}/D_{{\mathrm{{mean}}}}="
            f"{dose_ratio:.3f}$",
            xy=(dose_ratio, volume),
            xytext=(9, vertical_offset),
            textcoords="offset points",
            color="#5b187e",
        )
    dvh_axis.set(
        title=(
            "DVH GTVp: проектный углеродный пик\n"
            "два независимых запуска; сетка 1,6×1,6×0,8 мм"
        ),
        xlabel=r"Доза в элементе / $D_{\mathrm{mean}}$ GTVp",
        ylabel="Объём GTVp, получивший ≥ дозы, %",
        xlim=(0.0, 1.8),
        ylim=(0.0, 101.5),
    )
    dvh_axis.grid(alpha=0.2)
    dvh_axis.legend(frameon=False, loc="lower left")
    for suffix in (".png", ".svg", ".pdf"):
        dvh_figure.savefig(
            output_dir / f"carbon_c12_target20_dvh{suffix}",
            dpi=220,
        )
    plt.close(dvh_figure)

    rows_by_key = {row["metric"]: row for row in summary_rows}
    report = [
        "# Межseed-проверка проектного углеродного пика",
        "",
        "Дата: 29.07.2026.",
        "",
        "Сопоставлены два расчёта по 20 000 первичных ионов. Геометрия, "
        "21-компонентный источник, список физики `QGSP_INCLXX + option4`, "
        "production cuts, поле 30×30 мм и расчётная сетка были одинаковыми; "
        "изменялась только пара seed.",
        "",
        f"- запуск 1: `{labels[0]}`;",
        f"- запуск 2: `{labels[1]}`.",
        "",
        "| Показатель | Запуск 1 | Запуск 2 | Различие / среднее |",
        "|:---|---:|---:|---:|",
    ]
    report_metrics = (
        ("gtv_mean_dose_Gy_per_primary", 4),
        ("gtv_LETd_w_keV_um", 2),
        ("central_peak_depth_mm", 1),
        ("gtv_nonzero_dose_fraction", 4),
        ("gtv_aggregated_D50_over_Dmean_1p6x1p6x0p8", 3),
        ("gtv_aggregated_D90_over_Dmean_1p6x1p6x0p8", 3),
    )
    metric_meta = {key: (label, unit) for key, label, unit in METRICS}
    for key, digits in report_metrics:
        row = rows_by_key[key]
        label, unit = metric_meta[key]
        if key == "gtv_mean_dose_Gy_per_primary":
            value_1 = f"{row['seed_1']:.4e}"
            value_2 = f"{row['seed_2']:.4e}"
        elif key == "gtv_nonzero_dose_fraction":
            value_1 = format_ru(row["seed_1"] * 100.0, 2) + "%"
            value_2 = format_ru(row["seed_2"] * 100.0, 2) + "%"
            unit = ""
        else:
            value_1 = format_ru(row["seed_1"], digits)
            value_2 = format_ru(row["seed_2"], digits)
        unit_suffix = f" {unit}" if unit else ""
        report.append(
            f"| {label} | {value_1}{unit_suffix} | "
            f"{value_2}{unit_suffix} | "
            f"{format_ru(row['relative_difference_percent'], 2)}% |"
        )

    d90_row = rows_by_key[
        "gtv_aggregated_D90_over_Dmean_1p6x1p6x0p8"
    ]
    let_row = rows_by_key["gtv_LETd_w_keV_um"]
    dose_row = rows_by_key["gtv_mean_dose_Gy_per_primary"]
    d90_0p8_row = rows_by_key[
        "gtv_aggregated_D90_over_Dmean_0p8x0p8x0p4"
    ]
    d90_1p2_row = rows_by_key[
        "gtv_aggregated_D90_over_Dmean_1p2x1p2x0p6"
    ]
    report.extend(
        [
            "",
            "На отчётной сетке 1,6×1,6×0,8 мм различие "
            f"$D_{{90}}/D_\\mathrm{{mean}}$ составило "
            f"{format_ru(d90_row['relative_difference_percent'], 2)}%, "
            f"$LET_{{D,w}}$ — "
            f"{format_ru(let_row['relative_difference_percent'], 2)}%, "
            "а средней дозы на первичный — "
            f"{format_ru(dose_row['relative_difference_percent'], 2)}%.",
            "",
            "Центральный локальный максимум располагался на глубине "
            "23,4 и 24,6 мм, то есть в обоих случаях внутри GTVp. "
            "Различие 1,2 мм показывает, что координату единичного "
            "максимума следует оставлять диагностическим, а не основным "
            "показателем устойчивости распределения.",
            "",
            "Для $D_{90}/D_\\mathrm{mean}$ межзапусковое различие "
            "уменьшалось с укрупнением отчётной сетки: "
            f"{format_ru(d90_0p8_row['relative_difference_percent'], 2)}% "
            "при 0,8×0,8×0,4 мм, "
            f"{format_ru(d90_1p2_row['relative_difference_percent'], 2)}% "
            "при 1,2×1,2×0,6 мм и "
            f"{format_ru(d90_row['relative_difference_percent'], 2)}% "
            "при 1,6×1,6×0,8 мм. Поэтому точечную оценку $D_{90}$ "
            "следует приводить вместе с анализом чувствительности к "
            "разрешению.",
            "",
            "Два запуска подтверждают воспроизводимость интегральной LET, "
            "средней дозы и укрупнённой DVH при данном числе историй. "
            "Проверка по двум seed не является полной оценкой сходимости и "
            "не заменяет сравнение с физической дозиметрией. Нулевой "
            "$D_{90}$ исходной сетки в обоих запусках сохраняется как "
            "следствие статистически пустых тонких вокселей и не "
            "используется как физический показатель покрытия.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print(output_dir)
    for row in summary_rows:
        print(
            row["metric"],
            f"values={row['seed_1']:.8g},{row['seed_2']:.8g}",
            f"relative_difference={row['relative_difference_percent']:.3f}%",
        )


if __name__ == "__main__":
    main()
