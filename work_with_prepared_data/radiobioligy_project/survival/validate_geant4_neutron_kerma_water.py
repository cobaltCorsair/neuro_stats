"""Convert track-length neutron fluence to water kerma and validate it.

The direct Geant4 energy-deposition dose is not used to select or fit any
fluence-to-kerma coefficient.  Coefficients are reconstructed from the
published Ritts-Solomito-Stevens H and O contributions and interpolated at
track-length-weighted group mean energies from an independent 1000-history
calibration run.
"""

from __future__ import annotations

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

from work_with_prepared_data.radiobioligy_project.survival import (  # noqa: E402
    analyze_geant4_neutron_track_length as track_analysis,
)
from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (  # noqa: E402
    MatplotlibConfigurator,
)


TASK_DIR = Path(r"C:\dev\dissertation\task4_5")
RUN_ROOT = TASK_DIR / "scoring_v2_neutron_water_track_length"
CALIBRATION_RUN = RUN_ROOT / "kerma_smoke_1k"
VALIDATION_RUNS = ("seedA_10k", "seedB_10k", "seedC_10k")
COEFFICIENT_SOURCE = (
    TASK_DIR
    / "source_models"
    / "neutron_kerma_ritts1969_selected.csv"
)
OUTPUT = (
    TASK_DIR
    / "outputs"
    / "geant4_livermore_20260724"
    / "neutron_water_kerma_validation"
)

WATER_H_MASS_FRACTION = 0.11189834407236524
WATER_O_MASS_FRACTION = 1.0 - WATER_H_MASS_FRACTION
STANDARD_MAN_H_MASS_FRACTION = 0.10
STANDARD_MAN_O_MASS_FRACTION = 0.60
ERG_PER_G_TO_PGY = 1.0e8
KEV_TO_J = 1.602176634e-16
WATER_VOXEL_MASS_KG = 1.0e-6
CORE_DEPTH_SLICE = slice(10, 90)


def read_coefficient_source() -> dict[str, np.ndarray]:
    rows: list[dict[str, float]] = []
    with COEFFICIENT_SOURCE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for raw in csv.DictReader(handle):
            row = {key: float(value) for key, value in raw.items()}
            h_element = (
                row["hydrogen_contribution_erg_g_per_n_cm2"]
                / STANDARD_MAN_H_MASS_FRACTION
            )
            o_element = (
                row["oxygen_contribution_erg_g_per_n_cm2"]
                / STANDARD_MAN_O_MASS_FRACTION
            )
            row["water_kerma_pGy_cm2"] = (
                WATER_H_MASS_FRACTION * h_element
                + WATER_O_MASS_FRACTION * o_element
            ) * ERG_PER_G_TO_PGY
            rows.append(row)
    rows.sort(key=lambda row: row["energy_MeV"])
    return {
        key: np.asarray([row[key] for row in rows], dtype=float)
        for key in rows[0]
    }


def interpolation(
    energy_mev: float,
    source: dict[str, np.ndarray],
) -> float:
    energy_grid = source["energy_MeV"]
    if energy_mev < energy_grid[0] or energy_mev > energy_grid[-1]:
        raise ValueError(
            f"Energy {energy_mev:g} MeV lies outside coefficient table "
            f"{energy_grid[0]:g}-{energy_grid[-1]:g} MeV"
        )
    return float(
        np.interp(
            energy_mev,
            energy_grid,
            source["water_kerma_pGy_cm2"],
        )
    )


def run_histories(run_dir: Path) -> int:
    metadata = json.loads(
        (run_dir / "run_metadata.json").read_text(
            encoding="utf-8-sig"
        )
    )
    return int(metadata["histories"])


def group_calibration(
    source: dict[str, np.ndarray],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for label in track_analysis.GROUP_LABELS:
        path = track_analysis.find_track_map(CALIBRATION_RUN, label)
        if path is None:
            if label == "ge20MeV":
                rows.append(
                    {
                        "energy_group": label,
                        "mean_energy_MeV": np.nan,
                        "water_kerma_pGy_cm2": np.nan,
                        "calibration_track_length_mm": 0.0,
                    }
                )
                continue
            raise RuntimeError(f"Missing calibration group {label}")
        parsed = track_analysis.parse_voxel_map(path)
        length = float(np.sum(parsed["trackLength_mm"]))
        energy_length = float(
            np.sum(parsed["trackEnergyLength_MeV_mm"])
        )
        if length <= 0.0:
            mean_energy = np.nan
            coefficient = np.nan
        else:
            mean_energy = energy_length / length
            coefficient = interpolation(mean_energy, source)
        rows.append(
            {
                "energy_group": label,
                "mean_energy_MeV": mean_energy,
                "water_kerma_pGy_cm2": coefficient,
                "calibration_track_length_mm": length,
            }
        )
    return rows


def direct_dose_depth(
    run_dir: Path,
    histories: int,
) -> np.ndarray:
    main = track_analysis.parse_voxel_map(
        track_analysis.find_main_map(run_dir)
    )
    ids = np.asarray(main["voxel_id"], dtype=np.int64)
    energy_kev = np.asarray(main["depEnergy_keV"], dtype=float)
    selected = track_analysis.core_mask(ids)
    depth_energy_kev = np.bincount(
        ids[selected] % track_analysis.NX,
        weights=energy_kev[selected],
        minlength=track_analysis.NX,
    )
    n_voxels = track_analysis.core_voxels_per_depth()
    return (
        depth_energy_kev
        * KEV_TO_J
        / WATER_VOXEL_MASS_KG
        / histories
        / n_voxels
    )


def kerma_depth(
    run_dir: Path,
    histories: int,
    calibration: list[dict[str, float | str]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    total = np.zeros(track_analysis.NX, dtype=float)
    components: dict[str, np.ndarray] = {}
    coefficient_by_group = {
        str(row["energy_group"]): float(row["water_kerma_pGy_cm2"])
        for row in calibration
    }
    for label in track_analysis.GROUP_LABELS:
        path = track_analysis.find_track_map(run_dir, label)
        if path is None:
            components[label] = np.zeros(track_analysis.NX, dtype=float)
            continue
        parsed = track_analysis.parse_voxel_map(path)
        coefficient = coefficient_by_group[label]
        if not np.isfinite(coefficient):
            if np.sum(parsed["trackLength_mm"]) > 0.0:
                raise RuntimeError(
                    f"Non-zero fluence in unsupported group {label}"
                )
            components[label] = np.zeros(track_analysis.NX, dtype=float)
            continue
        fluence_mm2 = track_analysis.depth_fluence(parsed, histories)
        # 1 mm^-2 = 100 cm^-2; 1 pGy = 1e-12 Gy.
        component = fluence_mm2 * coefficient * 1.0e-10
        components[label] = component
        total += component
    return total, components


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_plot(
    direct_profiles: list[np.ndarray],
    kerma_profiles: list[np.ndarray],
) -> None:
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    depth = np.arange(track_analysis.NX, dtype=float) + 0.5
    direct_stack = np.vstack(direct_profiles)
    kerma_stack = np.vstack(kerma_profiles)
    direct_mean = np.mean(direct_stack, axis=0)
    kerma_mean = np.mean(kerma_stack, axis=0)
    ratio_stack = kerma_stack / np.maximum(
        direct_stack, np.finfo(float).tiny
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.4, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": (1.35, 0.8)},
    )
    axes[0].fill_between(
        depth,
        np.min(direct_stack, axis=0) * 1.0e12,
        np.max(direct_stack, axis=0) * 1.0e12,
        color="#4c78a8",
        alpha=0.18,
    )
    axes[0].plot(
        depth,
        direct_mean * 1.0e12,
        color="#2b6ea6",
        linewidth=2.5,
        label="прямой энерговклад Geant4",
    )
    axes[0].fill_between(
        depth,
        np.min(kerma_stack, axis=0) * 1.0e12,
        np.max(kerma_stack, axis=0) * 1.0e12,
        color="#e68613",
        alpha=0.18,
    )
    axes[0].plot(
        depth,
        kerma_mean * 1.0e12,
        color="#d87500",
        linewidth=2.5,
        label="трековый флюенс × табличная керма",
    )
    axes[0].set_ylabel(
        "пГр на первичный нейтрон",
        fontsize=18,
        labelpad=12,
    )
    axes[0].set_title(
        "Проверка преобразования нейтронного флюенса в воде",
        pad=16,
    )
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, ncol=2, loc="upper center")

    axes[1].fill_between(
        depth,
        np.min(ratio_stack, axis=0),
        np.max(ratio_stack, axis=0),
        color="#8c6bb1",
        alpha=0.20,
        label="диапазон трёх seed",
    )
    axes[1].plot(
        depth,
        np.mean(ratio_stack, axis=0),
        color="#6a3d9a",
        linewidth=2.1,
        label="среднее отношение",
    )
    axes[1].axhline(1.0, color="#202020", linestyle="--", linewidth=1.4)
    axes[1].axvspan(10, 90, color="#000000", alpha=0.035)
    axes[1].set_xlabel("глубина в воде, мм")
    axes[1].set_ylabel("K / D", fontsize=18, labelpad=12)
    axes[1].set_ylim(0.75, 1.25)
    axes[1].grid(alpha=0.20)
    axes[1].legend(frameon=False, ncol=2, loc="upper center")
    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        bottom=0.10,
        top=0.92,
        hspace=0.10,
    )
    fig.savefig(
        OUTPUT / "neutron_water_kerma_validation.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT / "neutron_water_kerma_validation.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    configurator.restore_original_styles()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = read_coefficient_source()
    calibration = group_calibration(source)

    coefficient_rows: list[dict[str, object]] = []
    for row in calibration:
        coefficient_rows.append(
            {
                **row,
                "coefficient_source": (
                    "Ritts et al. 1969; H/O mixture rule for water"
                ),
            }
        )
    write_csv(
        OUTPUT / "neutron_kerma_group_coefficients.csv",
        coefficient_rows,
    )

    summaries: list[dict[str, object]] = []
    depth_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    direct_profiles: list[np.ndarray] = []
    kerma_profiles: list[np.ndarray] = []
    for run_name in VALIDATION_RUNS:
        run_dir = RUN_ROOT / run_name
        histories = run_histories(run_dir)
        direct = direct_dose_depth(run_dir, histories)
        kerma, components = kerma_depth(
            run_dir, histories, calibration
        )
        direct_profiles.append(direct)
        kerma_profiles.append(kerma)
        central_direct = float(np.mean(direct[CORE_DEPTH_SLICE]))
        central_kerma = float(np.mean(kerma[CORE_DEPTH_SLICE]))
        summaries.append(
            {
                "run_tag": run_name,
                "histories": histories,
                "central_10_90mm_direct_dose_Gy_per_primary": (
                    central_direct
                ),
                "central_10_90mm_kerma_Gy_per_primary": central_kerma,
                "kerma_to_direct_ratio": central_kerma / central_direct,
                "relative_difference_percent": (
                    100.0 * (central_kerma / central_direct - 1.0)
                ),
            }
        )
        component_central = {
            label: float(
                np.mean(component[CORE_DEPTH_SLICE])
            )
            for label, component in components.items()
        }
        component_total = sum(component_central.values())
        for row in calibration:
            label = str(row["energy_group"])
            contribution_rows.append(
                {
                    "run_tag": run_name,
                    "energy_group": label,
                    "mean_energy_MeV": row["mean_energy_MeV"],
                    "water_kerma_pGy_cm2": row[
                        "water_kerma_pGy_cm2"
                    ],
                    "central_kerma_Gy_per_primary": (
                        component_central[label]
                    ),
                    "central_kerma_fraction_percent": (
                        100.0
                        * component_central[label]
                        / component_total
                        if component_total > 0.0
                        else np.nan
                    ),
                }
            )
        for index in range(track_analysis.NX):
            depth_rows.append(
                {
                    "run_tag": run_name,
                    "depth_mm": index + 0.5,
                    "direct_dose_Gy_per_primary": direct[index],
                    "kerma_Gy_per_primary": kerma[index],
                    "kerma_to_direct_ratio": (
                        kerma[index] / direct[index]
                        if direct[index] > 0.0
                        else np.nan
                    ),
                }
            )

    write_csv(
        OUTPUT / "neutron_water_kerma_validation_summary.csv",
        summaries,
    )
    write_csv(
        OUTPUT / "neutron_water_kerma_depth_profiles.csv",
        depth_rows,
    )
    write_csv(
        OUTPUT / "neutron_water_kerma_group_contributions.csv",
        contribution_rows,
    )
    build_plot(direct_profiles, kerma_profiles)

    ratios = np.asarray(
        [float(row["kerma_to_direct_ratio"]) for row in summaries]
    )
    direct_means = np.asarray(
        [
            float(
                row[
                    "central_10_90mm_direct_dose_Gy_per_primary"
                ]
            )
            for row in summaries
        ]
    )
    kerma_means = np.asarray(
        [
            float(
                row["central_10_90mm_kerma_Gy_per_primary"]
            )
            for row in summaries
        ]
    )
    report = f"""# Проверка преобразования нейтронного флюенса в керму воды

Дата расчёта: 2026-07-28.

## Метод

Трековый оцениватель сохранял для каждого энергетического интервала
`sum(W*L)` и `sum(W*L*E)`. Флюенс вычисляли как
`Phi_g = sum(W*L)_g / (N*V)`, а среднюю энергию группы — как
`Ebar_g = sum(W*L*E)_g / sum(W*L)_g`.

Керму воды рассчитывали независимо от прямого энерговклада:
`K = sum_g Phi_g * k_water(Ebar_g)`. Коэффициенты восстанавливали
из опубликованных вкладов H и O Ritts–Solomito–Stevens с использованием
массовой аддитивности для воды. Прямой энерговклад Geant4 не участвовал
ни в выборе, ни в настройке коэффициентов.

Для сравнения использованы три независимых запуска по 10 000 первичных
нейтронов. Основная область сравнения — центральные 10–90 мм водного
фантома; первые и последние 10 мм исключены из основной метрики из-за
нарушения приближения равновесия заряженных частиц у границ.

## Энергии и коэффициенты

| Группа | Средняя энергия, МэВ | k воды, пГр·см² |
|---|---:|---:|
"""
    for row in calibration:
        energy = float(row["mean_energy_MeV"])
        coefficient = float(row["water_kerma_pGy_cm2"])
        if np.isfinite(energy) and np.isfinite(coefficient):
            report += (
                f"| {row['energy_group']} | {energy:.4f} | "
                f"{coefficient:.3f} |\n"
            )
        else:
            report += f"| {row['energy_group']} | — | — |\n"

    report += f"""
## Результат проверки

В центральных 10–90 мм отношение кермы по трековому флюенсу к прямой
поглощённой дозе составило от {np.min(ratios):.4f} до
{np.max(ratios):.4f}; среднее по трём seed — {np.mean(ratios):.4f}.
Иными словами, расхождение отдельных запусков находилось в пределах
{100.0 * np.max(np.abs(ratios - 1.0)):.2f}%.

Средняя прямая доза составляла
{np.mean(direct_means) * 1.0e12:.3f} пГр на первичный нейтрон,
трековая оценка кермы — {np.mean(kerma_means) * 1.0e12:.3f} пГр.
Межseed-вариабельность трековой оценки была
{100.0 * np.std(kerma_means, ddof=1) / np.mean(kerma_means):.2f}%,
прямого энерговклада —
{100.0 * np.std(direct_means, ddof=1) / np.mean(direct_means):.2f}%.

## Интерпретация и ограничения

Проверка подтверждает вычислительную корректность цепочки
`трековая длина → флюенс → керма` в однородной воде для данного спектра
14,7 МэВ. Она не является экспериментальной валидацией абсолютной дозы:
геометрия, спектр и прямой энерговклад получены в той же транспортной
модели Geant4. Численные коэффициенты взяты из открытой исторической
таблицы; для формального текста их следует связать с более поздним
согласованным набором ICRU 63 / Chadwick et al. (1999). Группа ≥20 МэВ
в текущих запусках пуста и поэтому не экстраполировалась.
"""
    (OUTPUT / "NEUTRON_WATER_KERMA_VALIDATION_2026-07-28.md").write_text(
        report,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
