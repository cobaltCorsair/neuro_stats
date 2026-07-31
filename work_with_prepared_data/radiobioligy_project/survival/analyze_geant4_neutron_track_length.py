"""Analyse the neutron track-length fluence benchmark in water.

The Geant4 scorer persists weighted neutron step length W*L per voxel and
per event.  Fluence is calculated as sum(W*L)/voxel volume.  No
fluence-to-kerma coefficient is applied in this script: the resulting maps
are a low-variance fluence precursor, not absorbed-dose maps.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import struct
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


TASK_DIR = Path(r"C:\dev\dissertation\task4_5")
RUN_ROOT = TASK_DIR / "scoring_v2_neutron_water_track_length"
OUTPUT = (
    TASK_DIR
    / "outputs"
    / "geant4_livermore_20260724"
    / "neutron_water_track_length"
)
NX, NY, NZ = 100, 80, 80
VOXEL_VOLUME_MM3 = 1.0
FIELD_RADIUS_MM = 17.3
GROUP_LABELS = (
    "lt0p4MeV",
    "0p4to1MeV",
    "1to5MeV",
    "5to10MeV",
    "10to20MeV",
    "ge20MeV",
)
GROUP_DISPLAY = {
    "lt0p4MeV": "<0,4 МэВ",
    "0p4to1MeV": "0,4–1 МэВ",
    "1to5MeV": "1–5 МэВ",
    "5to10MeV": "5–10 МэВ",
    "10to20MeV": "10–20 МэВ",
    "ge20MeV": "≥20 МэВ",
}
GROUP_COLORS = {
    "lt0p4MeV": "#542788",
    "0p4to1MeV": "#8073ac",
    "1to5MeV": "#3288bd",
    "5to10MeV": "#66c2a5",
    "10to20MeV": "#e6ab02",
    "ge20MeV": "#d53e4f",
}


def read_varint(handle) -> int:
    value = 0
    shift = 0
    while True:
        raw = handle.read(1)
        if not raw:
            raise EOFError
        byte = raw[0]
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value
        shift += 7
        if shift > 63:
            raise ValueError("Invalid protobuf varint")


def skip_field(handle, wire: int) -> None:
    if wire == 0:
        read_varint(handle)
    elif wire == 1:
        handle.seek(8, 1)
    elif wire == 2:
        handle.seek(read_varint(handle), 1)
    elif wire == 5:
        handle.seek(4, 1)
    else:
        raise ValueError(f"Unsupported protobuf wire type: {wire}")


def parse_voxel_map(path: Path) -> dict[str, np.ndarray | int]:
    voxel_ids: list[int] = []
    dep_energy: list[float] = []
    track_length: list[float] = []
    track_length2: list[float] = []
    track_energy_length: list[float] = []
    n_events = 0

    with path.open("rb") as handle:
        while True:
            try:
                tag = read_varint(handle)
            except EOFError:
                break
            field, wire = tag >> 3, tag & 7
            if field != 1 or wire != 2:
                skip_field(handle, wire)
                continue

            entry_length = read_varint(handle)
            entry_end = handle.tell() + entry_length
            voxel_id: int | None = None
            energy = 0.0
            length = 0.0
            length2 = 0.0
            energy_length = 0.0
            while handle.tell() < entry_end:
                entry_tag = read_varint(handle)
                entry_field, entry_wire = entry_tag >> 3, entry_tag & 7
                if entry_field == 1 and entry_wire == 0:
                    voxel_id = read_varint(handle)
                elif entry_field == 2 and entry_wire == 2:
                    value_length = read_varint(handle)
                    value_end = handle.tell() + value_length
                    while handle.tell() < value_end:
                        value_tag = read_varint(handle)
                        value_field, value_wire = value_tag >> 3, value_tag & 7
                        if value_wire == 1 and value_field in (2, 13, 14, 15):
                            value = struct.unpack("<d", handle.read(8))[0]
                            if value_field == 2:
                                energy = value
                            elif value_field == 13:
                                length = value
                            elif value_field == 14:
                                length2 = value
                            else:
                                energy_length = value
                        elif value_field == 6 and value_wire == 0:
                            n_events = max(n_events, read_varint(handle))
                        else:
                            skip_field(handle, value_wire)
                    handle.seek(value_end)
                else:
                    skip_field(handle, entry_wire)
            handle.seek(entry_end)
            if voxel_id is None:
                continue
            if energy > 0.0 or length > 0.0:
                voxel_ids.append(voxel_id)
                dep_energy.append(energy)
                track_length.append(length)
                track_length2.append(length2)
                track_energy_length.append(energy_length)

    return {
        "voxel_id": np.asarray(voxel_ids, dtype=np.int64),
        "depEnergy_keV": np.asarray(dep_energy, dtype=float),
        "trackLength_mm": np.asarray(track_length, dtype=float),
        "trackLength2_mm2": np.asarray(track_length2, dtype=float),
        "trackEnergyLength_MeV_mm": np.asarray(
            track_energy_length, dtype=float
        ),
        "nEvents": n_events,
    }


def find_main_map(run_dir: Path) -> Path:
    matches = [
        path
        for path in run_dir.glob("vox_*_0")
        if "_neutron_track_length_" not in path.name
        and "_component_" not in path.name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one main map in {run_dir}, found {len(matches)}"
        )
    return matches[0]


def find_track_map(run_dir: Path, label: str) -> Path | None:
    matches = list(
        run_dir.glob(f"vox_*_neutron_track_length_{label}_0")
    )
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {label} map in {run_dir}, found {len(matches)}"
        )
    return matches[0]


def empty_map() -> dict[str, np.ndarray | int]:
    return {
        "voxel_id": np.asarray([], dtype=np.int64),
        "depEnergy_keV": np.asarray([], dtype=float),
        "trackLength_mm": np.asarray([], dtype=float),
        "trackLength2_mm2": np.asarray([], dtype=float),
        "trackEnergyLength_MeV_mm": np.asarray([], dtype=float),
        "nEvents": 0,
    }


def core_mask(voxel_ids: np.ndarray) -> np.ndarray:
    y = (voxel_ids // NX) % NY
    z = voxel_ids // (NX * NY)
    y_mm = y.astype(float) + 0.5 - NY / 2.0
    z_mm = z.astype(float) + 0.5 - NZ / 2.0
    return y_mm * y_mm + z_mm * z_mm <= FIELD_RADIUS_MM**2


def core_voxels_per_depth() -> int:
    y, z = np.meshgrid(
        np.arange(NY, dtype=float) + 0.5 - NY / 2.0,
        np.arange(NZ, dtype=float) + 0.5 - NZ / 2.0,
        indexing="ij",
    )
    return int(np.count_nonzero(y * y + z * z <= FIELD_RADIUS_MM**2))


def depth_fluence(
    parsed: dict[str, np.ndarray | int],
    histories: int,
) -> np.ndarray:
    ids = np.asarray(parsed["voxel_id"], dtype=np.int64)
    lengths = np.asarray(parsed["trackLength_mm"], dtype=float)
    selected = core_mask(ids)
    sums = np.bincount(
        ids[selected] % NX,
        weights=lengths[selected],
        minlength=NX,
    )
    volume_per_depth = core_voxels_per_depth() * VOXEL_VOLUME_MM3
    return sums / (histories * volume_per_depth)


def median_voxel_rse(
    parsed: dict[str, np.ndarray | int],
    histories: int,
    *,
    restrict_to_core: bool = False,
) -> float:
    ids = np.asarray(parsed["voxel_id"], dtype=np.int64)
    sums = np.asarray(parsed["trackLength_mm"], dtype=float)
    sums2 = np.asarray(parsed["trackLength2_mm2"], dtype=float)
    if restrict_to_core:
        selected = core_mask(ids)
        sums = sums[selected]
        sums2 = sums2[selected]
    if histories <= 1 or sums.size == 0:
        return math.nan
    residual = np.maximum(sums2 - sums * sums / histories, 0.0)
    variance_mean = residual / (histories * (histories - 1))
    mean = sums / histories
    valid = mean > 0.0
    if not np.any(valid):
        return math.nan
    return float(np.median(np.sqrt(variance_mean[valid]) / mean[valid]))


def analyse_run(
    run_dir: Path,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, np.ndarray],
]:
    metadata = json.loads(
        (run_dir / "run_metadata.json").read_text(encoding="utf-8-sig")
    )
    histories = int(metadata["histories"])
    main = parse_voxel_map(find_main_map(run_dir))
    all_path = find_track_map(run_dir, "all")
    if all_path is None:
        raise RuntimeError(f"Missing all-energy track map in {run_dir}")
    all_map = parse_voxel_map(all_path)
    groups: dict[str, dict[str, np.ndarray | int]] = {}
    for label in GROUP_LABELS:
        path = find_track_map(run_dir, label)
        groups[label] = parse_voxel_map(path) if path else empty_map()

    all_total = float(
        np.sum(np.asarray(all_map["trackLength_mm"], dtype=float))
    )
    group_totals = {
        label: float(
            np.sum(
                np.asarray(groups[label]["trackLength_mm"], dtype=float)
            )
        )
        for label in GROUP_LABELS
    }
    group_sum = sum(group_totals.values())
    core_profile_all = depth_fluence(all_map, histories)
    profiles = {
        "all": core_profile_all,
        **{
            label: depth_fluence(groups[label], histories)
            for label in GROUP_LABELS
        },
    }

    entrance = float(np.mean(core_profile_all[:10]))
    exit_value = float(np.mean(core_profile_all[-10:]))
    total_voxels = NX * NY * NZ
    total_core_voxels = core_voxels_per_depth() * NX
    track_core_nonzero = int(
        np.count_nonzero(
            core_mask(np.asarray(all_map["voxel_id"], dtype=np.int64))
        )
    )
    deposit_core_nonzero = int(
        np.count_nonzero(
            core_mask(np.asarray(main["voxel_id"], dtype=np.int64))
        )
    )
    summary = {
        "run_tag": metadata["run_tag"],
        "histories": histories,
        "seed1": metadata["seed1"],
        "seed2": metadata["seed2"],
        "all_track_length_mm": all_total,
        "mean_track_length_mm_per_primary": all_total / histories,
        "mean_fluence_full_phantom_mm-2_per_primary": (
            all_total
            / (histories * total_voxels * VOXEL_VOLUME_MM3)
        ),
        "entrance_core_fluence_mm-2_per_primary": entrance,
        "exit_core_fluence_mm-2_per_primary": exit_value,
        "exit_to_entrance_ratio": exit_value / entrance,
        "track_nonzero_voxels": len(all_map["voxel_id"]),
        "deposit_nonzero_voxels": len(main["voxel_id"]),
        "track_coverage_percent": (
            100.0 * len(all_map["voxel_id"]) / total_voxels
        ),
        "deposit_coverage_percent": (
            100.0 * len(main["voxel_id"]) / total_voxels
        ),
        "coverage_multiplier": (
            len(all_map["voxel_id"]) / max(len(main["voxel_id"]), 1)
        ),
        "track_core_coverage_percent": (
            100.0 * track_core_nonzero / total_core_voxels
        ),
        "deposit_core_coverage_percent": (
            100.0 * deposit_core_nonzero / total_core_voxels
        ),
        "core_coverage_multiplier": (
            track_core_nonzero / max(deposit_core_nonzero, 1)
        ),
        "energy_group_closure_percent": 100.0 * group_sum / all_total,
        "median_nonzero_voxel_RSE": median_voxel_rse(
            all_map, histories
        ),
        "median_core_voxel_RSE": median_voxel_rse(
            all_map,
            histories,
            restrict_to_core=True,
        ),
    }
    group_rows = []
    for label in GROUP_LABELS:
        group_rows.append(
            {
                "run_tag": metadata["run_tag"],
                "histories": histories,
                "energy_group": label,
                "display_label": GROUP_DISPLAY[label],
                "track_length_mm": group_totals[label],
                "track_length_fraction_percent": (
                    100.0 * group_totals[label] / all_total
                ),
                "nonzero_voxels": len(groups[label]["voxel_id"]),
            }
        )
    return summary, group_rows, profiles


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_depth_plot(
    summaries: list[dict[str, object]],
    profiles_by_run: dict[str, dict[str, np.ndarray]],
) -> None:
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    depth = np.arange(NX, dtype=float) + 0.5
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.5, 10.5),
        sharex=True,
        gridspec_kw={"height_ratios": (1.0, 1.2)},
    )

    all_profiles = []
    for row in summaries:
        tag = str(row["run_tag"])
        profile = profiles_by_run[tag]["all"]
        all_profiles.append(profile)
        axes[0].plot(
            depth,
            profile,
            linewidth=1.2,
            alpha=0.65,
            label=tag.replace("seed", "seed ").replace("_10k", ""),
        )
    pooled_all = np.mean(np.vstack(all_profiles), axis=0)
    axes[0].plot(
        depth,
        pooled_all,
        color="#202020",
        linewidth=3.0,
        label="среднее трёх seed",
    )
    axes[0].axhline(
        1.0 / (math.pi * FIELD_RADIUS_MM**2),
        color="#b2182b",
        linestyle="--",
        linewidth=1.7,
        label=r"$1/(\pi r^2)$",
    )
    axes[0].set_ylabel(r"$\Phi$, мм$^{-2}$/нейтрон")
    axes[0].set_title(
        "Трековый оцениватель флюенса в центральном поле",
        pad=52,
    )
    axes[0].grid(alpha=0.22)
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=5,
        frameon=False,
        fontsize=13,
    )

    pooled_groups = {
        label: np.mean(
            np.vstack(
                [
                    profiles_by_run[str(row["run_tag"])][label]
                    for row in summaries
                ]
            ),
            axis=0,
        )
        for label in GROUP_LABELS
    }
    denominator = np.maximum(
        sum(pooled_groups.values()),
        np.finfo(float).tiny,
    )
    fractions = [
        100.0 * pooled_groups[label] / denominator
        for label in GROUP_LABELS
    ]
    axes[1].stackplot(
        depth,
        fractions,
        labels=[GROUP_DISPLAY[label] for label in GROUP_LABELS],
        colors=[GROUP_COLORS[label] for label in GROUP_LABELS],
        alpha=0.88,
    )
    axes[1].set_xlabel("глубина в воде, мм")
    axes[1].set_ylabel("доля трекового флюенса, %")
    axes[1].set_ylim(0.0, 100.0)
    axes[1].grid(alpha=0.18)
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        fontsize=13,
    )
    fig.tight_layout(rect=(0.035, 0.03, 1.0, 0.98))
    fig.savefig(
        OUTPUT / "neutron_track_length_depth_and_spectrum.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT / "neutron_track_length_depth_and_spectrum.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    configurator.restore_original_styles()


def build_coverage_plot(summaries: list[dict[str, object]]) -> None:
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    x = np.arange(len(summaries))
    width = 0.34
    track = [
        float(row["track_core_coverage_percent"]) for row in summaries
    ]
    deposit = [
        float(row["deposit_core_coverage_percent"]) for row in summaries
    ]
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    bars_track = ax.bar(
        x - width / 2,
        track,
        width,
        label="трековая длина нейтронов",
        color="#2b8cbe",
    )
    bars_deposit = ax.bar(
        x + width / 2,
        deposit,
        width,
        label="ненулевой энерговклад",
        color="#e6550d",
    )
    ax.bar_label(bars_track, fmt="%.1f%%", padding=4, fontsize=13)
    ax.bar_label(bars_deposit, fmt="%.1f%%", padding=4, fontsize=13)
    ax.set_xticks(
        x,
        [
            str(row["run_tag"]).replace("seed", "seed ").replace("_10k", "")
            for row in summaries
        ],
    )
    ax.set_ylabel("ненулевые воксели, %")
    ax.set_title("Пространственное заполнение в пределах поля")
    ax.set_ylim(
        0.0,
        max(track + deposit) * 1.22,
    )
    ax.grid(axis="y", alpha=0.22)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0.04, 0.03, 1.0, 1.0))
    fig.savefig(
        OUTPUT / "neutron_track_length_coverage.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT / "neutron_track_length_coverage.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    configurator.restore_original_styles()


def coefficient_of_variation(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size < 2 or np.mean(array) == 0.0:
        return math.nan
    return float(100.0 * np.std(array, ddof=1) / np.mean(array))


def write_report(
    summaries: list[dict[str, object]],
    group_rows: list[dict[str, object]],
) -> None:
    entrance = [
        float(row["entrance_core_fluence_mm-2_per_primary"])
        for row in summaries
    ]
    total_length = [
        float(row["mean_track_length_mm_per_primary"])
        for row in summaries
    ]
    coverage_ratios = [
        float(row["core_coverage_multiplier"]) for row in summaries
    ]
    closures = [
        float(row["energy_group_closure_percent"])
        for row in summaries
    ]
    summary_table = "\n".join(
        "| {run_tag} | {histories} | "
        "{entrance_core_fluence_mm-2_per_primary:.6g} | "
        "{exit_to_entrance_ratio:.3f} | "
        "{track_core_coverage_percent:.2f} | "
        "{deposit_core_coverage_percent:.2f} | "
        "{energy_group_closure_percent:.6f} |".format(**row)
        for row in summaries
    )
    pooled_group = {}
    for label in GROUP_LABELS:
        values = [
            float(row["track_length_fraction_percent"])
            for row in group_rows
            if row["energy_group"] == label
        ]
        pooled_group[label] = float(np.mean(values))
    group_table = "\n".join(
        f"| {GROUP_DISPLAY[label]} | {pooled_group[label]:.3f} |"
        for label in GROUP_LABELS
    )
    report = f"""# Нейтронный track-length benchmark в воде

Дата: 2026-07-28.

В каждом вокселе накапливалась сумма `W·L` для нейтронных шагов до
проверки локального энерговклада. Флюенс определяли как
`Σ(W·L)/V`. Это соответствует определению клеточного флюенса
`G4PSCellFlux` в документации Geant4. Сохранялись сумма и сумма квадратов
событийных значений, а также шесть энергетических групп.

| запуск | N | входной флюенс в поле, мм⁻²/нейтрон | выход/вход | track coverage в поле, % | deposit coverage в поле, % | замыкание групп, % |
|---|---:|---:|---:|---:|---:|---:|
{summary_table}

CV входного флюенса между seed составил
`{coefficient_of_variation(entrance):.3f}%`, CV средней трековой длины на
первичный нейтрон — `{coefficient_of_variation(total_length):.3f}%`.
В пределах номинального круглого поля трековый оцениватель заполнял в
среднем `{np.mean([float(row["track_core_coverage_percent"]) for row in summaries]):.3f}%`
вокселей, а прямой энерговклад —
`{np.mean([float(row["deposit_core_coverage_percent"]) for row in summaries]):.3f}%`.
Отношение числа ненулевых вокселей составило
`{np.mean(coverage_ratios):.2f}`. Медианная относительная стандартная
ошибка отдельного
1-мм вокселя внутри поля оставалась
`{100.0 * np.mean([float(row["median_core_voxel_RSE"]) for row in summaries]):.1f}%`;
поэтому текущая статистика достаточна для устойчивого глубинного профиля,
но ещё недостаточна для мелковоксельной DVH без укрупнения сетки или
увеличения числа историй. Замыкание суммы энергетических групп:
`{min(closures):.6f}–{max(closures):.6f}%`.

| энергетическая группа | средняя доля трековой длины, % |
|---|---:|
{group_table}

## Статус интерпретации

Полученная величина является флюенсом, а не поглощённой дозой. Перевод в
керму требует независимого набора энергетически и материал-специфических
коэффициентов либо отдельно верифицированной Geant4-калибровки. До этого
трековую карту можно использовать для проверки пространственной
сходимости и спектрального состава, но нельзя подменять ею DVH дозы.

Официальное определение Geant4:
https://geant4.web.cern.ch/documentation/dev/bfad_html/ForApplicationDevelopers/Detector/hit.html
"""
    (OUTPUT / "NEUTRON_TRACK_LENGTH_WATER_2026-07-28.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_dirs = []
    for path in sorted(RUN_ROOT.iterdir()):
        metadata_path = path / "run_metadata.json"
        if not path.is_dir() or not metadata_path.exists():
            continue
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8-sig")
        )
        has_main = any(
            "_neutron_track_length_" not in path.name
            and "_component_" not in path.name
            for path in path.glob("vox_*_0")
        )
        has_track = any(
            path.glob("vox_*_neutron_track_length_all_0")
        )
        if (
            int(metadata.get("histories", 0)) >= 10000
            and has_main
            and has_track
        ):
            run_dirs.append(path)
    if len(run_dirs) < 2:
        raise RuntimeError(
            f"Need at least two >=10k runs in {RUN_ROOT}; "
            f"found {len(run_dirs)}"
        )

    summaries: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    profiles_by_run: dict[str, dict[str, np.ndarray]] = {}
    for run_dir in run_dirs:
        summary, groups, profiles = analyse_run(run_dir)
        summaries.append(summary)
        group_rows.extend(groups)
        profiles_by_run[str(summary["run_tag"])] = profiles

    depth_rows: list[dict[str, object]] = []
    for summary in summaries:
        tag = str(summary["run_tag"])
        for ix in range(NX):
            depth_rows.append(
                {
                    "run_tag": tag,
                    "depth_mm": ix + 0.5,
                    **{
                        f"fluence_{label}_mm-2_per_primary": (
                            profiles_by_run[tag][label][ix]
                        )
                        for label in ("all", *GROUP_LABELS)
                    },
                }
            )

    write_csv(OUTPUT / "neutron_track_length_summary.csv", summaries)
    write_csv(
        OUTPUT / "neutron_track_length_energy_groups.csv", group_rows
    )
    write_csv(
        OUTPUT / "neutron_track_length_depth_profiles.csv", depth_rows
    )
    build_depth_plot(summaries, profiles_by_run)
    build_coverage_plot(summaries)
    write_report(summaries, group_rows)
    for row in summaries:
        print(
            "{run_tag}: entry={entrance_core_fluence_mm-2_per_primary:.6g} "
            "mm^-2/primary, exit/entry={exit_to_entrance_ratio:.3f}, "
            "core coverage={track_core_coverage_percent:.2f}% vs "
            "{deposit_core_coverage_percent:.2f}%, "
            "closure={energy_group_closure_percent:.6f}%".format(**row)
        )


if __name__ == "__main__":
    main()
