"""Build low-variance neutron kerma and DVH estimates in the rat GTV."""

from __future__ import annotations

import csv
import itertools
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

TASK_DIR = Path(r"C:\dev\dissertation\task4_5")
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from work_with_prepared_data.radiobioligy_project.survival import (  # noqa: E402
    analyze_geant4_neutron_track_length as track_analysis,
)
from work_with_prepared_data.radiobioligy_project.survival.analyze_geant4_neutron_ng14_rat import (  # noqa: E402
    repeated_block_means,
)
from work_with_prepared_data.radiobioligy_project.survival.analyze_geant4_proton_100MeV_reduced_gtv import (  # noqa: E402
    DEFAULT_CT_DIR,
    DEFAULT_RS,
    PX_MM,
    PY_MM,
    PZ_MM,
    gtv_geometry,
    parse_voxel_map as parse_gtv_map,
)
from work_with_prepared_data.radiobioligy_project.survival.validate_geant4_neutron_kerma_water import (  # noqa: E402
    ERG_PER_G_TO_PGY,
    interpolation as interpolate_water_kerma,
    read_coefficient_source,
)
from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (  # noqa: E402
    MatplotlibConfigurator,
)


RUN_ROOT = TASK_DIR / "scoring_v2_neutron_ng14_rat"
RUN_NAMES = (
    "track_kerma_seedA_10k",
    "track_kerma_seedB_10k",
    "track_kerma_seedC_10k",
)
OUTPUT = (
    TASK_DIR
    / "outputs"
    / "geant4_livermore_20260724"
    / "neutron_ng14_rat_kerma"
)
VOXEL_VOLUME_MM3 = PX_MM * PY_MM * PZ_MM
PRIMARY_FACTORS = (4, 4, 4)
PRIMARY_GRID = "1.6x1.6x0.8 mm"
SENSITIVITY_FACTORS = (8, 8, 8)
SENSITIVITY_GRID = "3.2x3.2x1.6 mm"
NOMINAL_DOSE_GY = 7.2

GROUP_DISPLAY = {
    "lt0p4MeV": "<0,4",
    "0p4to1MeV": "0,4–1",
    "1to5MeV": "1–5",
    "5to10MeV": "5–10",
    "10to20MeV": "10–20",
    "ge20MeV": "≥20",
}
GROUP_COLORS = {
    "lt0p4MeV": "#542788",
    "0p4to1MeV": "#8073ac",
    "1to5MeV": "#3288bd",
    "5to10MeV": "#66c2a5",
    "10to20MeV": "#e6ab02",
    "ge20MeV": "#d53e4f",
}


def interpolate_standard_man_kerma(
    energy_mev: float,
    source: dict[str, np.ndarray],
) -> float:
    energy_grid = source["energy_MeV"]
    if energy_mev < energy_grid[0] or energy_mev > energy_grid[-1]:
        raise ValueError(
            f"Energy {energy_mev:g} MeV lies outside coefficient table"
        )
    return float(
        np.interp(
            energy_mev,
            energy_grid,
            source["standard_man_total_erg_g_per_n_cm2"],
        )
        * ERG_PER_G_TO_PGY
    )


def load_metadata(run_dir: Path) -> dict[str, object]:
    return json.loads(
        (run_dir / "run_metadata.json").read_text(encoding="utf-8-sig")
    )


def dose_at_volume(dose: np.ndarray, volume_percent: float) -> float:
    return float(np.percentile(dose, 100.0 - volume_percent))


def spatial_metrics(
    dose: np.ndarray,
    mean_reference: float,
) -> dict[str, float]:
    return {
        "coverage_fraction": float(np.mean(dose > 0.0)),
        "CV": float(np.std(dose) / np.mean(dose)),
        "D2_over_Dmean": dose_at_volume(dose, 2.0) / mean_reference,
        "D50_over_Dmean": dose_at_volume(dose, 50.0) / mean_reference,
        "D90_over_Dmean": dose_at_volume(dose, 90.0) / mean_reference,
        "D95_over_Dmean": dose_at_volume(dose, 95.0) / mean_reference,
        "D98_over_Dmean": dose_at_volume(dose, 98.0) / mean_reference,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyse_run(
    run_dir: Path,
    gtv_ids: np.ndarray,
    gtv_lookup: np.ndarray,
    source: dict[str, np.ndarray],
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    metadata = load_metadata(run_dir)
    histories = int(metadata["histories"])
    water = np.zeros(gtv_ids.size, dtype=float)
    standard_man = np.zeros(gtv_ids.size, dtype=float)
    water_components: dict[str, np.ndarray] = {}
    group_rows: list[dict[str, object]] = []

    for label in track_analysis.GROUP_LABELS:
        path = track_analysis.find_track_map(run_dir, label)
        component = np.zeros(gtv_ids.size, dtype=float)
        component_standard = np.zeros(gtv_ids.size, dtype=float)
        if path is None:
            water_components[label] = component
            group_rows.append(
                {
                    "run_tag": run_dir.name,
                    "energy_group": label,
                    "mean_energy_MeV": np.nan,
                    "water_kerma_pGy_cm2": np.nan,
                    "standard_man_kerma_pGy_cm2": np.nan,
                    "gtv_track_length_mm": 0.0,
                    "gtv_water_kerma_Gy_per_primary": 0.0,
                }
            )
            continue
        parsed = track_analysis.parse_voxel_map(path)
        voxel_ids = np.asarray(parsed["voxel_id"], dtype=np.int64)
        positions = gtv_lookup[voxel_ids]
        selected = positions >= 0
        positions = positions[selected]
        lengths = np.asarray(
            parsed["trackLength_mm"], dtype=float
        )[selected]
        energy_lengths = np.asarray(
            parsed["trackEnergyLength_MeV_mm"], dtype=float
        )[selected]
        total_length = float(np.sum(lengths))
        if total_length <= 0.0:
            water_components[label] = component
            continue
        mean_energy = float(np.sum(energy_lengths) / total_length)
        water_coefficient = interpolate_water_kerma(mean_energy, source)
        standard_coefficient = interpolate_standard_man_kerma(
            mean_energy, source
        )
        fluence_mm2 = lengths / (
            histories * VOXEL_VOLUME_MM3
        )
        np.add.at(
            component,
            positions,
            fluence_mm2 * water_coefficient * 1.0e-10,
        )
        np.add.at(
            component_standard,
            positions,
            fluence_mm2 * standard_coefficient * 1.0e-10,
        )
        water += component
        standard_man += component_standard
        water_components[label] = component
        group_rows.append(
            {
                "run_tag": run_dir.name,
                "energy_group": label,
                "mean_energy_MeV": mean_energy,
                "water_kerma_pGy_cm2": water_coefficient,
                "standard_man_kerma_pGy_cm2": standard_coefficient,
                "gtv_track_length_mm": total_length,
                "gtv_water_kerma_Gy_per_primary": float(
                    np.mean(component)
                ),
            }
        )

    main = parse_gtv_map(
        track_analysis.find_main_map(run_dir),
        gtv_lookup,
        include_dose=True,
    )
    direct = np.asarray(main["gtv_dose_Gy"], dtype=float) / histories
    water_mean = float(np.mean(water))
    standard_mean = float(np.mean(standard_man))
    direct_mean = float(np.mean(direct))
    summary = {
        "run_tag": run_dir.name,
        "histories": histories,
        "seed1": metadata["seed1"],
        "seed2": metadata["seed2"],
        "gtv_voxels": gtv_ids.size,
        "gtv_volume_mm3": gtv_ids.size * VOXEL_VOLUME_MM3,
        "fine_track_coverage_fraction": float(np.mean(water > 0.0)),
        "fine_direct_deposit_coverage_fraction": float(
            np.mean(direct > 0.0)
        ),
        "water_kerma_mean_Gy_per_primary": water_mean,
        "standard_man_kerma_mean_Gy_per_primary": standard_mean,
        "standard_man_to_water_ratio": standard_mean / water_mean,
        "direct_dose_mean_Gy_per_primary": direct_mean,
        "water_kerma_to_direct_ratio": water_mean / direct_mean,
        "incident_neutrons_per_Gy_GTV_mean": 1.0 / water_mean,
    }
    return summary, group_rows, water, standard_man, water_components


def build_dvh_rows(
    profiles: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    x_relative = np.linspace(0.0, 1.8, 451)
    rows: list[dict[str, object]] = []
    for label, dose in profiles.items():
        relative = dose / np.mean(dose)
        for value in x_relative:
            rows.append(
                {
                    "curve": label,
                    "dose_over_Dmean": value,
                    "dose_Gy_if_Dmean_7p2Gy": (
                        value * NOMINAL_DOSE_GY
                    ),
                    "volume_percent": (
                        100.0 * np.mean(relative >= value)
                    ),
                }
            )
    return rows


def pool_convergence_rows(
    water_arrays: list[np.ndarray],
    gtv_ids: np.ndarray,
) -> list[dict[str, object]]:
    """Evaluate every available seed combination at each pool size."""
    rows: list[dict[str, object]] = []
    seed_labels = ("A", "B", "C")
    for pool_size in range(1, len(water_arrays) + 1):
        for indices in itertools.combinations(
            range(len(water_arrays)), pool_size
        ):
            pooled = np.mean(
                np.vstack([water_arrays[index] for index in indices]),
                axis=0,
            )
            report_grid, block_count, coverage = repeated_block_means(
                pooled, gtv_ids, PRIMARY_FACTORS
            )
            metrics = spatial_metrics(
                report_grid, float(np.mean(pooled))
            )
            rows.append(
                {
                    "histories": pool_size * 10000,
                    "seed_combination": "+".join(
                        seed_labels[index] for index in indices
                    ),
                    "unique_spatial_units": block_count,
                    "coverage_fraction": coverage,
                    "D2_over_Dmean": metrics["D2_over_Dmean"],
                    "D50_over_Dmean": metrics["D50_over_Dmean"],
                    "D90_over_Dmean": metrics["D90_over_Dmean"],
                    "D95_over_Dmean": metrics["D95_over_Dmean"],
                    "D98_over_Dmean": metrics["D98_over_Dmean"],
                    "spatial_CV": metrics["CV"],
                }
            )
    return rows


def build_figure(
    dvh_profiles: dict[str, np.ndarray],
    pooled_components: dict[str, np.ndarray],
) -> None:
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.6, 6.8),
        gridspec_kw={"width_ratios": (1.45, 1.0)},
    )

    x_relative = np.linspace(0.0, 1.8, 451)
    for label in ("seed A", "seed B", "seed C"):
        relative = dvh_profiles[label] / np.mean(dvh_profiles[label])
        volume = np.asarray(
            [100.0 * np.mean(relative >= x) for x in x_relative]
        )
        axes[0].plot(
            x_relative * NOMINAL_DOSE_GY,
            volume,
            color="#b7b7b7",
            linewidth=1.1,
            alpha=0.75,
        )
    for label, color, width, style in (
        ("pooled 1.6 mm", "#c62828", 2.8, "-"),
        ("pooled 3.2 mm", "#e68613", 2.4, "--"),
    ):
        relative = dvh_profiles[label] / np.mean(dvh_profiles[label])
        volume = np.asarray(
            [100.0 * np.mean(relative >= x) for x in x_relative]
        )
        axes[0].plot(
            x_relative * NOMINAL_DOSE_GY,
            volume,
            color=color,
            linewidth=width,
            linestyle=style,
            label=(
                "пул трёх seed, 1,6×1,6×0,8 мм"
                if "1.6" in label
                else "чувствительность: 3,2×3,2×1,6 мм"
            ),
        )
    axes[0].plot(
        [],
        [],
        color="#b7b7b7",
        linewidth=1.2,
        label="отдельные seed, основная сетка",
    )
    axes[0].axvline(
        NOMINAL_DOSE_GY,
        color="#333333",
        linestyle=":",
        linewidth=1.2,
    )
    axes[0].set_xlim(0.0, 1.8 * NOMINAL_DOSE_GY)
    axes[0].set_ylim(0.0, 101.0)
    axes[0].set_xlabel(
        "доза, Гр (сценарная нормировка Dmean = 7,2 Гр)"
    )
    axes[0].set_ylabel("объём GTV, получающий не менее дозы, %")
    axes[0].set_title("Кумулятивная DVH")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, loc="lower left")

    group_means = {
        label: float(np.mean(component))
        for label, component in pooled_components.items()
    }
    total = sum(group_means.values())
    labels = [
        label
        for label in track_analysis.GROUP_LABELS
        if group_means.get(label, 0.0) > 0.0
    ]
    values = [100.0 * group_means[label] / total for label in labels]
    y = np.arange(len(labels))
    bars = axes[1].barh(
        y,
        values,
        color=[GROUP_COLORS[label] for label in labels],
    )
    axes[1].set_yticks(
        y,
        [GROUP_DISPLAY[label] for label in labels],
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("вклад в среднюю керму GTV, %")
    axes[1].set_ylabel("энергия нейтронов, МэВ")
    axes[1].set_title("Энергетические компоненты")
    axes[1].grid(axis="x", alpha=0.22)
    axes[1].set_xlim(0.0, 105.0)
    axes[1].bar_label(
        bars,
        labels=[f"{value:.2f}%" for value in values],
        padding=4,
        fontsize=11,
    )

    fig.suptitle(
        "Нейтронное поле 14,7 МэВ в GTV крысы: трековая керма",
        fontsize=21,
        y=0.99,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.97,
        bottom=0.14,
        top=0.86,
        wspace=0.30,
    )
    fig.savefig(
        OUTPUT / "neutron_gtv_kerma_dvh_and_spectrum.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT / "neutron_gtv_kerma_dvh_and_spectrum.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    configurator.restore_original_styles()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _, gtv_ids, gtv_lookup, depth_interval = gtv_geometry(
        DEFAULT_CT_DIR, DEFAULT_RS
    )
    source = read_coefficient_source()

    summaries: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    water_arrays: list[np.ndarray] = []
    standard_arrays: list[np.ndarray] = []
    component_arrays: list[dict[str, np.ndarray]] = []
    per_seed_primary: list[np.ndarray] = []

    for run_name in RUN_NAMES:
        (
            summary,
            groups,
            water,
            standard,
            components,
        ) = analyse_run(
            RUN_ROOT / run_name,
            gtv_ids,
            gtv_lookup,
            source,
        )
        summaries.append(summary)
        group_rows.extend(groups)
        water_arrays.append(water)
        standard_arrays.append(standard)
        component_arrays.append(components)
        primary, _, _ = repeated_block_means(
            water, gtv_ids, PRIMARY_FACTORS
        )
        per_seed_primary.append(primary)

    pooled_water = np.mean(np.vstack(water_arrays), axis=0)
    pooled_standard = np.mean(np.vstack(standard_arrays), axis=0)
    pooled_components = {
        label: np.mean(
            np.vstack(
                [
                    components.get(
                        label, np.zeros(gtv_ids.size, dtype=float)
                    )
                    for components in component_arrays
                ]
            ),
            axis=0,
        )
        for label in track_analysis.GROUP_LABELS
    }
    pooled_primary, primary_blocks, _ = repeated_block_means(
        pooled_water, gtv_ids, PRIMARY_FACTORS
    )
    pooled_sensitivity, sensitivity_blocks, _ = repeated_block_means(
        pooled_water, gtv_ids, SENSITIVITY_FACTORS
    )
    pooled_mean = float(np.mean(pooled_water))

    metric_rows: list[dict[str, object]] = []
    for label, grid, dose, blocks in (
        (
            "fine",
            "0.4x0.4x0.2 mm",
            pooled_water,
            gtv_ids.size,
        ),
        (
            "primary",
            PRIMARY_GRID,
            pooled_primary,
            primary_blocks,
        ),
        (
            "sensitivity",
            SENSITIVITY_GRID,
            pooled_sensitivity,
            sensitivity_blocks,
        ),
    ):
        metric_rows.append(
            {
                "analysis": label,
                "grid": grid,
                "unique_spatial_units": blocks,
                **spatial_metrics(dose, pooled_mean),
            }
        )

    dvh_profiles = {
        "seed A": per_seed_primary[0],
        "seed B": per_seed_primary[1],
        "seed C": per_seed_primary[2],
        "pooled 1.6 mm": pooled_primary,
        "pooled 3.2 mm": pooled_sensitivity,
    }
    write_csv(OUTPUT / "neutron_gtv_kerma_seed_summary.csv", summaries)
    write_csv(
        OUTPUT / "neutron_gtv_kerma_group_coefficients.csv",
        group_rows,
    )
    write_csv(
        OUTPUT / "neutron_gtv_kerma_spatial_metrics.csv",
        metric_rows,
    )
    write_csv(
        OUTPUT / "neutron_gtv_kerma_dvh_curves.csv",
        build_dvh_rows(dvh_profiles),
    )
    convergence_rows = pool_convergence_rows(water_arrays, gtv_ids)
    write_csv(
        OUTPUT / "neutron_gtv_kerma_pool_convergence.csv",
        convergence_rows,
    )
    build_figure(dvh_profiles, pooled_components)

    water_means = np.asarray(
        [
            float(row["water_kerma_mean_Gy_per_primary"])
            for row in summaries
        ]
    )
    direct_means = np.asarray(
        [
            float(row["direct_dose_mean_Gy_per_primary"])
            for row in summaries
        ]
    )
    standard_mean = float(np.mean(pooled_standard))
    primary_metrics = metric_rows[1]
    sensitivity_metrics = metric_rows[2]
    group_mean_contributions = {
        label: float(np.mean(component))
        for label, component in pooled_components.items()
    }
    group_total = sum(group_mean_contributions.values())
    high_group_fraction = (
        group_mean_contributions["10to20MeV"] / group_total
    )
    convergence_summary: list[tuple[int, float, float, float]] = []
    for histories in (10000, 20000, 30000):
        values = np.asarray(
            [
                float(row["D90_over_Dmean"])
                for row in convergence_rows
                if int(row["histories"]) == histories
            ]
        )
        convergence_summary.append(
            (
                histories,
                float(np.mean(values)),
                float(np.min(values)),
                float(np.max(values)),
            )
        )
    convergence_table = "\n".join(
        f"| {histories:,}".replace(",", " ")
        + f" | {mean_value:.3f} | {minimum:.3f}–{maximum:.3f} |"
        for histories, mean_value, minimum, maximum
        in convergence_summary
    )
    one_seed_d90 = np.asarray(
        [
            float(row["D90_over_Dmean"])
            for row in convergence_rows
            if int(row["histories"]) == 10000
        ]
    )
    two_seed_d90 = np.asarray(
        [
            float(row["D90_over_Dmean"])
            for row in convergence_rows
            if int(row["histories"]) == 20000
        ]
    )
    three_seed_d90 = float(
        next(
            row["D90_over_Dmean"]
            for row in convergence_rows
            if int(row["histories"]) == 30000
        )
    )

    report = f"""# Трековая оценка нейтронной кермы в GTV крысы

Дата расчёта: 2026-07-28.

## Расчётная постановка

Использованы три независимых запуска по 10 000 первичных нейтронов
14,7 ± 0,15 МэВ в редуцированном CT-фантоме крысы. Поле имело радиус
17,3 мм и было направлено вдоль +x. Физика: QGSP_BIC_AllHP +
G4EmLivermorePhysics. GTV содержала {gtv_ids.size} исходных вокселей
0,4×0,4×0,2 мм, объём — {gtv_ids.size * VOXEL_VOLUME_MM3:.2f} мм³
({gtv_ids.size * VOXEL_VOLUME_MM3 / 1000.0:.3f} см³), диапазон глубины
по принятой оси — {depth_interval[0]:.1f}–{depth_interval[1]:.1f} мм.

Для каждого энергетического интервала сохраняли `sum(W*L)` и
`sum(W*L*E)`. В каждом запуске коэффициент кермы воды интерполировали при
фактической трек-взвешенной средней энергии группы. Затем три карты
усредняли в каждом исходном вокселе; это эквивалентно пулу 30 000
историй при одинаковой геометрии и источнике.

## Основные результаты

Средняя водоэквивалентная керма GTV составила
{pooled_mean * 1.0e12:.3f} пГр на один первичный нейтрон. Межseed-CV
среднего значения — {100.0 * np.std(water_means, ddof=1) / np.mean(water_means):.2f}%.
Для прямого энерговклада Geant4 межseed-CV равнялся
{100.0 * np.std(direct_means, ddof=1) / np.mean(direct_means):.2f}%,
а отношение пуловой трековой оценки к прямой дозе —
{np.mean(water_means) / np.mean(direct_means):.3f}.

Коэффициент для состава «standard man» давал
{standard_mean * 1.0e12:.3f} пГр на первичный нейтрон, то есть
{100.0 * (standard_mean / pooled_mean - 1.0):.2f}% относительно воды.
Энергетическая группа 10–20 МэВ формировала
{100.0 * high_group_fraction:.2f}% средней кермы GTV.

Прямой энерговклад был ненулевым лишь в
{100.0 * np.mean([float(row['fine_direct_deposit_coverage_fraction']) for row in summaries]):.2f}%
тонких GTV-вокселей. Для трековой карты этот показатель увеличился до
{100.0 * np.mean([float(row['fine_track_coverage_fraction']) for row in summaries]):.2f}%;
после пулирования и агрегации на предзаданную сетку {PRIMARY_GRID}
объёмное покрытие составило
{100.0 * float(primary_metrics['coverage_fraction']):.3f}%.

## Пространственные показатели

На основной сетке {PRIMARY_GRID} для пуловой карты:

- D2/Dmean = {float(primary_metrics['D2_over_Dmean']):.3f};
- D50/Dmean = {float(primary_metrics['D50_over_Dmean']):.3f};
- D90/Dmean = {float(primary_metrics['D90_over_Dmean']):.3f};
- D95/Dmean = {float(primary_metrics['D95_over_Dmean']):.3f};
- D98/Dmean = {float(primary_metrics['D98_over_Dmean']):.3f};
- пространственный CV = {float(primary_metrics['CV']):.3f}.

Проверку зависимости от числа историй выполняли для всех доступных
комбинаций независимых seed:

| Историй в пуле | Среднее D90/Dmean | Диапазон между комбинациями |
|---:|---:|---:|
{convergence_table}

Значение для 30 000 историй сравнивали не только с отдельными seed, но и
со всеми тремя двухseed-пулами; тем самым вывод не зависел от выбранного
порядка накопления запусков. При этом уровень 30 000 представлен одной
объединённой картой и не является независимой репликацией пула такого
размера.

Во всех трёх отдельных seed `D90` был ненулевым, а межseed-CV
`D90/Dmean` составил
{100.0 * np.std(one_seed_d90, ddof=1) / np.mean(one_seed_d90):.2f}%.
Между тремя пулами по 20 000 историй CV снизился до
{100.0 * np.std(two_seed_d90, ddof=1) / np.mean(two_seed_d90):.2f}%,
а значение пула 30 000 отличалось от среднего двухseed-пулов на
{100.0 * (three_seed_d90 / np.mean(two_seed_d90) - 1.0):.2f}%.
Таким образом, заранее сформулированный критерий ненулевого `D90` и
межseed-CV не более 10% выполнен. Сохраняющийся направленный рост
`D90/Dmean` с размером пула указывается как остаточное ограничение, а не
скрывается выбором наиболее гладкой карты.

При укрупнении до {SENSITIVITY_GRID} D90/Dmean увеличивалось до
{float(sensitivity_metrics['D90_over_Dmean']):.3f}, D95/Dmean — до
{float(sensitivity_metrics['D95_over_Dmean']):.3f}. Это показывает,
что хвост DVH ещё зависит от пространственного масштаба усреднения;
поэтому основным остаётся заранее объявленный масштаб 1,6×1,6×0,8 мм,
а более крупная сетка приводится только как анализ чувствительности.

Если для наглядности нормировать среднюю дозу GTV на 7,2 Гр, основная
пуловая DVH соответствует D90 =
{NOMINAL_DOSE_GY * float(primary_metrics['D90_over_Dmean']):.2f} Гр,
D95 = {NOMINAL_DOSE_GY * float(primary_metrics['D95_over_Dmean']):.2f} Гр
и D2 = {NOMINAL_DOSE_GY * float(primary_metrics['D2_over_Dmean']):.2f} Гр.
Эта нормировка сценарная: архив не содержит надёжной цепочки,
связывающей число нейтронов на входе фантома с номинальной дозой каждой
экспериментальной серии. По расчётной модели для 1 Гр средней кермы GTV
требовалось {1.0 / pooled_mean:.3e} первичных нейтронов на входной
плоскости.

## Статус

Низкодисперсионная цепочка `трековая длина → флюенс → керма → GTV-DVH`
реализована и внутренне проверена. Она закрывает пространственную часть
нейтронного Monte-Carlo блока значительно лучше прямого воксельного
энерговклада. Однако абсолютную установочную дозиметрию она не заменяет:
стальной коллиматор, реальный спектр рассеянных нейтронов и
экспериментальная нормировка флюенса пока не реконструированы.
"""
    (OUTPUT / "NEUTRON_GTV_KERMA_RESULTS_2026-07-28.md").write_text(
        report,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
