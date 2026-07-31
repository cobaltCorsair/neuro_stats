"""Compare 20-mm and 23-mm prospective C-12 target-width sources."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPORTING_SUFFIX = "1p6x1p6x0p8"
GTV_LOW_MM = 17.8
GTV_HIGH_MM = 34.6


def read_single_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, got {len(rows)}")
    return rows[0]


def read_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No data rows in {path}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in rows[0]
    }


def dose_at_volume(
    dose_ratio: np.ndarray,
    volume_percent: np.ndarray,
    requested_volume: float,
) -> float:
    below = np.flatnonzero(volume_percent <= requested_volume)
    if below.size == 0:
        return float(dose_ratio[-1])
    right = int(below[0])
    if right == 0:
        return float(dose_ratio[0])
    left = right - 1
    y0, y1 = volume_percent[left], volume_percent[right]
    x0, x1 = dose_ratio[left], dose_ratio[right]
    if y0 == y1:
        return float(x1)
    fraction = (requested_volume - y0) / (y1 - y0)
    return float(x0 + fraction * (x1 - x0))


def group_dvh(case_dirs: list[Path], common_x: np.ndarray) -> np.ndarray:
    curves = []
    for case_dir in case_dirs:
        columns = read_columns(
            case_dir / "gtv_analysis" / "through_normalised_dvh.csv"
        )
        curves.append(
            np.interp(
                common_x,
                columns["dose_over_gtv_mean"],
                columns[
                    f"volume_receiving_percent_{REPORTING_SUFFIX}"
                ],
            )
        )
    return np.vstack(curves)


def group_depth_profiles(
    case_dirs: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy_profiles = []
    let_profiles = []
    depth_reference: np.ndarray | None = None
    for case_dir in case_dirs:
        columns = read_columns(
            case_dir / "gtv_analysis" / "through_central_depth_profile.csv"
        )
        depth = columns["depth_mm_from_plus_y_surface"]
        if depth_reference is None:
            depth_reference = depth
        elif not np.array_equal(depth_reference, depth):
            raise ValueError("Depth grids differ between cases")
        energy = columns["central_energy_smoothed_keV"]
        gtv = (depth >= GTV_LOW_MM) & (depth <= GTV_HIGH_MM)
        energy_profiles.append(energy / np.mean(energy[gtv]))
        let_profiles.append(columns["central_LETd_w_smoothed_keV_um"])
    assert depth_reference is not None
    return (
        depth_reference,
        np.vstack(energy_profiles),
        np.vstack(let_profiles),
    )


def metric_values(
    summaries: list[dict[str, str]],
    key: str,
) -> np.ndarray:
    return np.asarray([float(summary[key]) for summary in summaries])


def percent_change(old: float, new: float) -> float:
    return (new - old) / old * 100.0


def format_ru(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-case", type=Path, action="append", required=True)
    parser.add_argument("--new-case", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.old_case) != 2 or len(args.new_case) != 2:
        parser.error("Specify exactly two --old-case and two --new-case paths")

    old_cases = [path.resolve() for path in args.old_case]
    new_cases = [path.resolve() for path in args.new_case]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    old_summaries = [
        read_single_row(path / "gtv_analysis" / "gtv_summary.csv")
        for path in old_cases
    ]
    new_summaries = [
        read_single_row(path / "gtv_analysis" / "gtv_summary.csv")
        for path in new_cases
    ]

    common_x = np.linspace(0.0, 2.2, 441)
    old_dvhs = group_dvh(old_cases, common_x)
    new_dvhs = group_dvh(new_cases, common_x)
    old_dvh_mean = np.mean(old_dvhs, axis=0)
    new_dvh_mean = np.mean(new_dvhs, axis=0)

    old_depth, old_energy, old_let = group_depth_profiles(old_cases)
    new_depth, new_energy, new_let = group_depth_profiles(new_cases)
    if not np.array_equal(old_depth, new_depth):
        raise ValueError("Old and new depth grids differ")
    depth = old_depth
    gtv_depth = (depth >= GTV_LOW_MM) & (depth <= GTV_HIGH_MM)

    summary_metric_keys = {
        "D2_over_Dmean": (
            f"gtv_aggregated_D2_over_Dmean_{REPORTING_SUFFIX}"
        ),
        "D50_over_Dmean": (
            f"gtv_aggregated_D50_over_Dmean_{REPORTING_SUFFIX}"
        ),
        "D90_over_Dmean": (
            f"gtv_aggregated_D90_over_Dmean_{REPORTING_SUFFIX}"
        ),
        "D95_over_Dmean": (
            f"gtv_aggregated_D95_over_Dmean_{REPORTING_SUFFIX}"
        ),
        "D98_over_Dmean": (
            f"gtv_aggregated_D98_over_Dmean_{REPORTING_SUFFIX}"
        ),
        "HI98": f"gtv_aggregated_HI98_{REPORTING_SUFFIX}",
        "LETd_w_keV_um": "gtv_LETd_w_keV_um",
        "nonzero_fraction": (
            f"gtv_aggregated_nonzero_volume_fraction_{REPORTING_SUFFIX}"
        ),
    }
    comparison_rows = []
    metric_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, key in summary_metric_keys.items():
        old_values = metric_values(old_summaries, key)
        new_values = metric_values(new_summaries, key)
        metric_arrays[label] = old_values, new_values
        old_mean = float(np.mean(old_values))
        new_mean = float(np.mean(new_values))
        comparison_rows.append(
            {
                "metric": label,
                "old_seed_1": old_values[0],
                "old_seed_2": old_values[1],
                "old_mean": old_mean,
                "old_sample_sd": float(np.std(old_values, ddof=1)),
                "new_seed_1": new_values[0],
                "new_seed_2": new_values[1],
                "new_mean": new_mean,
                "new_sample_sd": float(np.std(new_values, ddof=1)),
                "relative_change_percent": percent_change(
                    old_mean, new_mean
                ),
            }
        )

    old_d80 = np.asarray(
        [dose_at_volume(common_x, curve, 80.0) for curve in old_dvhs]
    )
    new_d80 = np.asarray(
        [dose_at_volume(common_x, curve, 80.0) for curve in new_dvhs]
    )
    comparison_rows.append(
        {
            "metric": "D80_over_Dmean",
            "old_seed_1": old_d80[0],
            "old_seed_2": old_d80[1],
            "old_mean": float(np.mean(old_d80)),
            "old_sample_sd": float(np.std(old_d80, ddof=1)),
            "new_seed_1": new_d80[0],
            "new_seed_2": new_d80[1],
            "new_mean": float(np.mean(new_d80)),
            "new_sample_sd": float(np.std(new_d80, ddof=1)),
            "relative_change_percent": percent_change(
                float(np.mean(old_d80)), float(np.mean(new_d80))
            ),
        }
    )

    with (output_dir / "target_width_comparison_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(comparison_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    curve_rows = []
    for index, dose_ratio in enumerate(common_x):
        curve_rows.append(
            {
                "dose_over_gtv_mean": dose_ratio,
                "target20_mean_volume_percent": old_dvh_mean[index],
                "target20_min_volume_percent": np.min(
                    old_dvhs[:, index]
                ),
                "target20_max_volume_percent": np.max(
                    old_dvhs[:, index]
                ),
                "target23_mean_volume_percent": new_dvh_mean[index],
                "target23_min_volume_percent": np.min(
                    new_dvhs[:, index]
                ),
                "target23_max_volume_percent": np.max(
                    new_dvhs[:, index]
                ),
            }
        )
    with (output_dir / "target_width_dvh_curves.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    old_colour = "#7a1fa2"
    new_colour = "#1f8a8a"
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
        3, 1, figsize=(9.5, 11.5), constrained_layout=True
    )
    ax_dvh, ax_depth, ax_let = axes

    for curves, mean_curve, colour, label in (
        (old_dvhs, old_dvh_mean, old_colour, "20 мм; 21 энергия"),
        (new_dvhs, new_dvh_mean, new_colour, "23 мм; 24 энергии"),
    ):
        ax_dvh.fill_between(
            common_x,
            np.min(curves, axis=0),
            np.max(curves, axis=0),
            color=colour,
            alpha=0.13,
        )
        ax_dvh.plot(
            common_x,
            mean_curve,
            color=colour,
            linewidth=2.6,
            label=label,
        )
        for volume, marker in ((90.0, "o"), (98.0, "s")):
            dose_ratio = dose_at_volume(common_x, mean_curve, volume)
            ax_dvh.scatter(
                [dose_ratio],
                [volume],
                color=colour,
                marker=marker,
                s=38,
                zorder=4,
            )
    ax_dvh.axhline(90.0, color="#777777", linestyle=":", linewidth=0.8)
    ax_dvh.axhline(98.0, color="#777777", linestyle=":", linewidth=0.8)
    ax_dvh.set(
        title=(
            "DVH GTVp: исходная и расширенная ширина пика\n"
            "среднее и диапазон двух seed; сетка 1,6×1,6×0,8 мм"
        ),
        xlabel=r"Доза в элементе / $D_{\mathrm{mean}}$ GTVp",
        ylabel="Объём GTVp, получивший ≥ дозы, %",
        xlim=(0.0, 1.8),
        ylim=(0.0, 101.5),
    )
    ax_dvh.grid(alpha=0.2)
    ax_dvh.legend(frameon=False, loc="lower left")

    for profiles, colour, label in (
        (old_energy, old_colour, "20 мм"),
        (new_energy, new_colour, "23 мм"),
    ):
        ax_depth.fill_between(
            depth,
            np.min(profiles, axis=0),
            np.max(profiles, axis=0),
            color=colour,
            alpha=0.13,
        )
        ax_depth.plot(
            depth,
            np.mean(profiles, axis=0),
            color=colour,
            linewidth=2.4,
            label=label,
        )
    ax_depth.axvspan(
        GTV_LOW_MM,
        GTV_HIGH_MM,
        color="#e31a1c",
        alpha=0.08,
        label="аксиальный интервал GTVp",
    )
    ax_depth.axhline(1.0, color="#777777", linewidth=0.8)
    ax_depth.set(
        title="Центральный энерговклад по глубине",
        xlabel="Глубина от входной границы +y, мм",
        ylabel="Энерговклад / среднее в глубинном интервале GTVp",
        xlim=(12.0, 44.0),
        ylim=(0.0, 1.35),
    )
    ax_depth.grid(alpha=0.2)
    ax_depth.legend(frameon=False, loc="upper right")

    let_ranges = {}
    for let_profiles, energy_profiles, colour, label, key in (
        (old_let, old_energy, old_colour, "20 мм", "target20"),
        (new_let, new_energy, new_colour, "23 мм", "target23"),
    ):
        relative_energy = energy_profiles / np.max(
            energy_profiles, axis=1, keepdims=True
        )
        reliable_depth = np.mean(relative_energy, axis=0) >= 0.05
        finite_depth = np.any(np.isfinite(let_profiles), axis=0)
        full_mean_let = np.full(depth.shape, np.nan, dtype=float)
        full_mean_let[finite_depth] = np.nanmean(
            let_profiles[:, finite_depth], axis=0
        )
        min_let = np.full(depth.shape, np.nan, dtype=float)
        max_let = np.full(depth.shape, np.nan, dtype=float)
        min_let[reliable_depth] = np.nanmin(
            let_profiles[:, reliable_depth], axis=0
        )
        max_let[reliable_depth] = np.nanmax(
            let_profiles[:, reliable_depth], axis=0
        )
        ax_let.fill_between(
            depth,
            min_let,
            max_let,
            color=colour,
            alpha=0.13,
        )
        ax_let.plot(
            depth,
            full_mean_let,
            color=colour,
            linewidth=1.4,
            linestyle="--",
            alpha=0.45,
        )
        ax_let.plot(
            depth,
            np.where(reliable_depth, full_mean_let, np.nan),
            color=colour,
            linewidth=2.4,
            label=label,
        )
        range_values = full_mean_let[
            gtv_depth & reliable_depth & np.isfinite(full_mean_let)
        ]
        let_ranges[key] = (
            float(np.min(range_values)),
            float(np.max(range_values)),
            float(np.mean(range_values)),
        )
    ax_let.axvspan(
        GTV_LOW_MM,
        GTV_HIGH_MM,
        color="#e31a1c",
        alpha=0.08,
        label="аксиальный интервал GTVp",
    )
    ax_let.set(
        title=(
            r"Дозо-взвешенная $LET_{D,w}$ центрального профиля"
            "\nполоса — диапазон двух seed"
        ),
        xlabel="Глубина от входной границы +y, мм",
        ylabel=r"$LET_{D,w}$, кэВ/мкм",
        xlim=(12.0, 44.0),
    )
    ax_let.grid(alpha=0.2)
    ax_let.text(
        0.02,
        0.97,
        (
            "Диапазон по глубине внутри GTVp:\n"
            f"20 мм: {let_ranges['target20'][0]:.1f}–"
            f"{let_ranges['target20'][1]:.1f} кэВ/мкм\n"
            f"23 мм: {let_ranges['target23'][0]:.1f}–"
            f"{let_ranges['target23'][1]:.1f} кэВ/мкм"
        ),
        transform=ax_let.transAxes,
        ha="left",
        va="top",
    )
    ax_let.text(
        0.02,
        0.03,
        (
            "Штриховая линия: энерговклад <5% максимума;\n"
            "LET в этом хвосте показана диагностически"
        ),
        transform=ax_let.transAxes,
        ha="left",
        va="bottom",
        color="#555555",
    )
    ax_let.legend(frameon=False, loc="upper right")

    figure.suptitle(
        "Влияние расширения проектного углеродного пика",
        fontsize=14,
    )
    for suffix in (".png", ".svg", ".pdf"):
        figure.savefig(
            output_dir / f"carbon_c12_target20_vs_target23{suffix}",
            dpi=220,
        )
    plt.close(figure)

    new_dvh_figure, new_dvh_axis = plt.subplots(
        1, 1, figsize=(8.8, 6.6), constrained_layout=True
    )
    new_dvh_axis.fill_between(
        common_x,
        np.min(new_dvhs, axis=0),
        np.max(new_dvhs, axis=0),
        color=new_colour,
        alpha=0.16,
        label="диапазон двух seed",
    )
    new_dvh_axis.plot(
        common_x,
        new_dvh_mean,
        color=new_colour,
        linewidth=3.0,
        label="средняя DVH",
    )
    for volume, marker, offset in (
        (98.0, "s", -17),
        (90.0, "o", -9),
        (50.0, "D", 7),
    ):
        dose_ratio = dose_at_volume(common_x, new_dvh_mean, volume)
        new_dvh_axis.axhline(
            volume, color="#777777", linestyle=":", linewidth=0.8
        )
        new_dvh_axis.scatter(
            [dose_ratio],
            [volume],
            marker=marker,
            color=new_colour,
            s=42,
            zorder=4,
        )
        new_dvh_axis.annotate(
            rf"$D_{{{int(volume)}}}/D_{{\mathrm{{mean}}}}="
            f"{dose_ratio:.3f}$",
            xy=(dose_ratio, volume),
            xytext=(9, offset),
            textcoords="offset points",
            color=new_colour,
        )
    new_dvh_axis.set(
        title=(
            "DVH GTVp: расширенный проектный углеродный пик\n"
            "24 энергии; среднее и диапазон двух seed"
        ),
        xlabel=r"Доза в элементе / $D_{\mathrm{mean}}$ GTVp",
        ylabel="Объём GTVp, получивший ≥ дозы, %",
        xlim=(0.0, 1.8),
        ylim=(0.0, 101.5),
    )
    new_hi98_mean = float(
        np.mean(metric_arrays["HI98"][1])
    )
    new_dvh_axis.text(
        0.98,
        0.96,
        (
            r"$HI_{98}=(D_2-D_{98})/D_{50}$"
            f"\n= {new_hi98_mean:.3f}"
        ),
        transform=new_dvh_axis.transAxes,
        ha="right",
        va="top",
        color=new_colour,
    )
    new_dvh_axis.grid(alpha=0.2)
    new_dvh_axis.legend(frameon=False, loc="lower left")
    for suffix in (".png", ".svg", ".pdf"):
        new_dvh_figure.savefig(
            output_dir / f"carbon_c12_target23_dvh{suffix}",
            dpi=220,
        )
    plt.close(new_dvh_figure)

    rows_by_metric = {row["metric"]: row for row in comparison_rows}
    report = [
        "# Сравнение проектных углеродных пиков шириной 20 и 23 мм",
        "",
        "Дата: 29.07.2026.",
        "",
        "Сопоставлены по два независимых запуска на 20 000 первичных "
        "ионов. Для расширенного варианта к прежнему 21-компонентному "
        "базису добавлены энергии 140,0; 1110,3 и 1136,6 МэВ на ядро "
        "$^{12}$C с водными максимумами 0,5; 22,5 и 23,5 мм.",
        "",
        "Отчётная DVH рассчитана на сетке 1,6×1,6×0,8 мм. "
        "$HI_{98}$ определён как $(D_2-D_{98})/D_{50}$; меньшее "
        "значение соответствует более однородному распределению.",
        "",
        "| Показатель | 20 мм, среднее | 23 мм, среднее | Изменение |",
        "|:---|---:|---:|---:|",
    ]
    report_order = (
        ("D80_over_Dmean", r"$D_{80}/D_\mathrm{mean}$"),
        ("D90_over_Dmean", r"$D_{90}/D_\mathrm{mean}$"),
        ("D95_over_Dmean", r"$D_{95}/D_\mathrm{mean}$"),
        ("D98_over_Dmean", r"$D_{98}/D_\mathrm{mean}$"),
        ("D2_over_Dmean", r"$D_2/D_\mathrm{mean}$"),
        ("HI98", r"$HI_{98}$"),
        ("LETd_w_keV_um", r"$LET_{D,w}$ GTVp, кэВ/мкм"),
    )
    for key, label in report_order:
        row = rows_by_metric[key]
        report.append(
            f"| {label} | {format_ru(row['old_mean'], 3)} | "
            f"{format_ru(row['new_mean'], 3)} | "
            f"{format_ru(row['relative_change_percent'], 2)}% |"
        )
    report.extend(
        [
            "",
            "Расширение практически не изменило $D_{80}$ и $D_{90}$, "
            "но повысило средний $D_{98}/D_\\mathrm{mean}$ и снизило "
            "$HI_{98}$. Следовательно, добавочные энергии улучшили "
            "крайний низкодозовый хвост, но не устранили общую "
            "неоднородность DVH.",
            "",
            "Средняя интегральная $LET_{D,w}$ GTVp снизилась, поэтому "
            "23-мм вариант нельзя автоматически объявлять лучшим: он "
            "реализует компромисс между покрытием крайних 2% объёма и "
            "уровнем LET.",
            "",
            "Диапазон средней по двум seed глубинной LET внутри "
            "аксиального интервала GTVp:",
            "",
            f"- 20-мм источник: {format_ru(let_ranges['target20'][0], 1)}–"
            f"{format_ru(let_ranges['target20'][1], 1)} кэВ/мкм;",
            f"- 23-мм источник: {format_ru(let_ranges['target23'][0], 1)}–"
            f"{format_ru(let_ranges['target23'][1], 1)} кэВ/мкм.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print(output_dir)
    for row in comparison_rows:
        print(
            row["metric"],
            f"old={row['old_mean']:.6g}",
            f"new={row['new_mean']:.6g}",
            f"change={row['relative_change_percent']:.3f}%",
        )
    print("LET depth ranges:", let_ranges)


if __name__ == "__main__":
    main()
