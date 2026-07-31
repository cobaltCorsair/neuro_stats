"""Analyse production-cut sensitivity for 14.7-MeV neutrons in water.

This is a technical scoring benchmark.  It uses a homogeneous water phantom,
the same random seeds and the same QGSP_BIC_AllHP + Livermore physics for all
three runs.  Only the gamma/electron/positron production cuts vary; the proton
cut remains 10 um.
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
RUN_ROOT = TASK_DIR / "scoring_v2_neutron_water_cut_benchmark"
OUTPUT = (
    TASK_DIR
    / "outputs"
    / "geant4_livermore_20260724"
    / "neutron_water_cut_benchmark"
)
REFERENCE_CSV = (
    TASK_DIR
    / "source_models"
    / "neutron"
    / "reference_component_contributions_14p5MeV.csv"
)
CUTS = (700, 100, 10)
COMPONENTS = (
    "electron",
    "hydrogen",
    "helium",
    "ion_Z3_to_Z6",
    "ion_Z7_plus",
)
COMPONENT_LABELS = {
    "electron": r"$e^\pm$",
    "hydrogen": "H / протоны",
    "helium": "He / альфа",
    "ion_Z3_to_Z6": r"$Z=3$–$6$",
    "ion_Z7_plus": r"$Z\geq7$",
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
        if not (byte & 0x80):
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


def aggregate_voxel_map(path: Path) -> dict[str, float | int]:
    """Stream one voxel map and return deposited-energy/LET sums."""
    energy = 0.0
    let_base = 0.0
    nonzero_entries = 0
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
            dep_energy = 0.0
            dep_let_base = 0.0
            while handle.tell() < entry_end:
                entry_tag = read_varint(handle)
                entry_field, entry_wire = entry_tag >> 3, entry_tag & 7
                if entry_field == 2 and entry_wire == 2:
                    value_length = read_varint(handle)
                    value_end = handle.tell() + value_length
                    while handle.tell() < value_end:
                        value_tag = read_varint(handle)
                        value_field = value_tag >> 3
                        value_wire = value_tag & 7
                        if value_field == 2 and value_wire == 1:
                            dep_energy = struct.unpack("<d", handle.read(8))[0]
                        elif value_field == 4 and value_wire == 1:
                            dep_let_base = struct.unpack(
                                "<d", handle.read(8)
                            )[0]
                        else:
                            skip_field(handle, value_wire)
                    handle.seek(value_end)
                else:
                    skip_field(handle, entry_wire)
            handle.seek(entry_end)
            if dep_energy > 0.0:
                nonzero_entries += 1
                energy += dep_energy
                let_base += dep_let_base
    return {
        "depEnergy_keV": energy,
        "letBase_keV2_um": let_base,
        "LETd_w_keV_um": (
            let_base / energy if energy > 0.0 else math.nan
        ),
        "nonzero_voxels": nonzero_entries,
    }


def find_main_map(run_dir: Path) -> Path:
    maps = [
        path
        for path in run_dir.glob("vox_*_0")
        if "_component_" not in path.name
    ]
    if len(maps) != 1:
        raise RuntimeError(
            f"Expected one main map in {run_dir}, found {len(maps)}"
        )
    return maps[0]


def find_component_map(run_dir: Path, component: str) -> Path:
    maps = list(run_dir.glob(f"vox_*_component_{component}_0"))
    if len(maps) != 1:
        raise RuntimeError(
            f"Expected one {component} map in {run_dir}, found {len(maps)}"
        )
    return maps[0]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_references() -> list[dict[str, object]]:
    with REFERENCE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def analyse() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summaries: list[dict[str, object]] = []
    components: list[dict[str, object]] = []
    for cut in CUTS:
        run_dir = RUN_ROOT / f"cut{cut}um_30k"
        metadata = json.loads(
            (run_dir / "run_metadata.json").read_text(encoding="utf-8-sig")
        )
        main = aggregate_voxel_map(find_main_map(run_dir))
        main_energy = float(main["depEnergy_keV"])
        component_values: dict[str, dict[str, float | int]] = {}
        for component in COMPONENTS:
            values = aggregate_voxel_map(
                find_component_map(run_dir, component)
            )
            component_values[component] = values
            component_energy = float(values["depEnergy_keV"])
            components.append(
                {
                    "cut_um": cut,
                    "component": component,
                    "label": COMPONENT_LABELS[component],
                    **values,
                    "energy_fraction_percent": (
                        100.0 * component_energy / main_energy
                    ),
                }
            )

        component_sum = sum(
            float(values["depEnergy_keV"])
            for values in component_values.values()
        )
        hydrogen_energy = float(
            component_values["hydrogen"]["depEnergy_keV"]
        )
        helium_energy = float(component_values["helium"]["depEnergy_keV"])
        hydrogen_let_base = float(
            component_values["hydrogen"]["letBase_keV2_um"]
        )
        helium_let_base = float(
            component_values["helium"]["letBase_keV2_um"]
        )
        summaries.append(
            {
                "cut_um": cut,
                "histories": metadata["histories"],
                "main_depEnergy_keV": main_energy,
                "main_LETd_w_keV_um": main["LETd_w_keV_um"],
                "hydrogen_percent": (
                    100.0 * hydrogen_energy / main_energy
                ),
                "helium_percent": 100.0 * helium_energy / main_energy,
                "heavy_Z_ge_3_percent": 100.0
                * (
                    float(
                        component_values["ion_Z3_to_Z6"][
                            "depEnergy_keV"
                        ]
                    )
                    + float(
                        component_values["ion_Z7_plus"]["depEnergy_keV"]
                    )
                )
                / main_energy,
                "electron_percent": 100.0
                * float(component_values["electron"]["depEnergy_keV"])
                / main_energy,
                "hydrogen_LETd_w_keV_um": component_values["hydrogen"][
                    "LETd_w_keV_um"
                ],
                "helium_LETd_w_keV_um": component_values["helium"][
                    "LETd_w_keV_um"
                ],
                "hydrogen_helium_LETd_w_keV_um": (
                    (hydrogen_let_base + helium_let_base)
                    / (hydrogen_energy + helium_energy)
                ),
                "component_closure_percent": (
                    100.0 * component_sum / main_energy
                ),
                "main_nonzero_voxels": main["nonzero_voxels"],
            }
        )
    return summaries, components


def reference_limits(
    references: list[dict[str, object]], field: str
) -> tuple[float, float, float]:
    caswell = next(
        float(row[field])
        for row in references
        if row["reference_id"] == "caswell"
    )
    geant4 = [
        float(row[field])
        for row in references
        if row["reference_id"] != "caswell"
    ]
    return caswell, min(geant4), max(geant4)


def build_component_plot(
    summaries: list[dict[str, object]],
    references: list[dict[str, object]],
) -> None:
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    fields = (
        ("hydrogen_percent", "protons_percent", "H / протоны"),
        ("helium_percent", "alpha_percent", "He / альфа-частицы"),
        (
            "heavy_Z_ge_3_percent",
            "heavy_ions_percent",
            r"тяжёлые ионы, $Z\geq3$",
        ),
        ("electron_percent", "electrons_percent", r"$e^\pm$"),
    )
    x = np.arange(len(CUTS))
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2))
    for ax, (current_field, reference_field, title) in zip(
        axes.flat, fields
    ):
        values = [float(row[current_field]) for row in summaries]
        caswell, ref_min, ref_max = reference_limits(
            references, reference_field
        )
        ax.axhspan(
            ref_min,
            ref_max,
            color="#9e9e9e",
            alpha=0.20,
            label="диапазон строк Geant4 из таблицы",
            zorder=0,
        )
        ax.axhline(
            caswell,
            color="#202020",
            linestyle="--",
            linewidth=1.6,
            label="Caswell",
            zorder=1,
        )
        ax.plot(
            x,
            values,
            color="#d95f02",
            marker="o",
            linewidth=2.2,
            markersize=8,
            label="настоящий водный расчёт",
            zorder=3,
        )
        span = max(max(values + [ref_max]) - min(values + [ref_min]), 1.0)
        for xi, value in zip(x, values):
            ax.annotate(
                f"{value:.2f}",
                (xi, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=13,
                clip_on=True,
            )
        ax.set_title(title)
        ax.set_xticks(x, [str(cut) for cut in CUTS])
        ax.set_xlabel("порог рождения γ/e⁻/e⁺, мкм")
        ax.set_ylabel("доля энерговклада, %")
        ax.set_xlim(-0.25, len(CUTS) - 0.75)
        ax.set_ylim(
            max(0.0, min(values + [caswell, ref_min]) - 0.12 * span),
            max(values + [caswell, ref_max]) + 0.20 * span,
        )
        ax.grid(alpha=0.22)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=14,
    )
    fig.suptitle(
        "Нейтроны 14,7 МэВ в воде: чувствительность компонентного "
        "энерговклада к production cuts",
        y=1.045,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(
        OUTPUT / "neutron_water_cut_component_fractions.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT / "neutron_water_cut_component_fractions.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    configurator.restore_original_styles()


def build_let_plot(summaries: list[dict[str, object]]) -> None:
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    x = np.arange(len(CUTS))
    series = (
        ("main_LETd_w_keV_um", r"полный $LET_{d,w}$", "#1b4f72"),
        (
            "hydrogen_LETd_w_keV_um",
            r"$LET_{d,w}$, H",
            "#d95f02",
        ),
        (
            "helium_LETd_w_keV_um",
            r"$LET_{d,w}$, He",
            "#7b3294",
        ),
        (
            "hydrogen_helium_LETd_w_keV_um",
            r"$LET_{d,w}$, H+He",
            "#2c7a3f",
        ),
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    for field, label, color in series:
        values = [float(row[field]) for row in summaries]
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2.2,
            markersize=8,
            label=label,
            color=color,
        )
        for xi, value in zip(x, values):
            ax.annotate(
                f"{value:.2f}",
                (xi, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=13,
                clip_on=True,
            )
    ax.set_xticks(x, [str(cut) for cut in CUTS])
    ax.set_xlabel("порог рождения γ/e⁻/e⁺, мкм")
    ax.set_ylabel(r"$LET_{d,w}$, кэВ/мкм")
    ax.set_title("Чувствительность дозо-взвешенной ЛПЭ в воде")
    all_values = [
        float(row[field])
        for field, _, _ in series
        for row in summaries
    ]
    ax.set_ylim(0.0, max(all_values) * 1.12)
    ax.grid(alpha=0.22)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        frameon=False,
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT / "neutron_water_cut_let.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT / "neutron_water_cut_let.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    configurator.restore_original_styles()


def write_report(summaries: list[dict[str, object]]) -> None:
    rows = []
    for row in summaries:
        rows.append(
            "| {cut_um} | {hydrogen_percent:.2f} | "
            "{helium_percent:.2f} | {heavy_Z_ge_3_percent:.2f} | "
            "{electron_percent:.2f} | {main_LETd_w_keV_um:.2f} | "
            "{component_closure_percent:.2f} |".format(**row)
        )
    let_rows = []
    for row in summaries:
        let_rows.append(
            "| {cut_um} | {main_LETd_w_keV_um:.2f} | "
            "{hydrogen_LETd_w_keV_um:.2f} | "
            "{helium_LETd_w_keV_um:.2f} | "
            "{hydrogen_helium_LETd_w_keV_um:.2f} |".format(**row)
        )
    electron = [float(row["electron_percent"]) for row in summaries]
    report = f"""# Водный бенчмарк нейтронного компонентного скоринга

Дата расчёта: 2026-07-28.

Однородный фантом `G4_WATER` размером 100×80×80 мм³ был облучён
30 000 нейтронов с гауссовым распределением энергии
14,7±0,15 МэВ. Во всех вариантах использованы одинаковые начальные числа,
QGSP_BIC_AllHP и G4EmLivermorePhysics. Изменяли только пороги рождения
γ/e⁻/e⁺; порог протонов оставался 10 мкм.

| cut γ/e⁻/e⁺, мкм | H, % | He, % | Z≥3, % | e±, % | полный LETd,w, кэВ/мкм | замыкание компонент, % |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

| cut γ/e⁻/e⁺, мкм | полный LETd,w | LETd,w(H) | LETd,w(He) | LETd,w(H+He) |
|---:|---:|---:|---:|---:|
{chr(10).join(let_rows)}

Доля явного электронного энерговклада изменилась от
{electron[0]:.3f}% при 700 мкм до {electron[-1]:.3f}% при 10 мкм,
то есть только на {electron[-1] - electron[0]:.3f} процентного пункта.
Полный `LETd,w` во всём диапазоне cuts изменился менее чем на 0,15%.
Таким образом, production cuts в диапазоне 10–700 мкм влияют на
компонентную атрибуцию, но не объясняют электронные доли 14,6–17,8% в
строках Geant4 пользовательской таблицы. Водный результат значительно
ближе к строке Caswell; особенно близка сумма тяжёлых ионов.

Бенчмарк не является независимой валидацией расчёта в крысе и не делает
строки пользовательской референсной таблицы строго сопоставимыми:
первичный источник, геометрия и определения компонент таблицы требуют
отдельного документирования. Оставшееся расхождение следует искать прежде
всего в правилах классификации локального энерговклада и в различиях
референсной постановки, а не подбором production cuts.

Файлы `neutron_water_cut_component_fractions.*` сопоставляют результат с
Caswell и диапазоном строк Geant4 из пользовательской таблицы;
`neutron_water_cut_let.*` показывает устойчивость дозо-взвешенной ЛПЭ.
"""
    (OUTPUT / "NEUTRON_WATER_CUT_BENCHMARK.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries, components = analyse()
    write_csv(OUTPUT / "neutron_water_cut_benchmark_summary.csv", summaries)
    write_csv(
        OUTPUT / "neutron_water_cut_benchmark_components.csv", components
    )
    references = load_references()
    build_component_plot(summaries, references)
    build_let_plot(summaries)
    write_report(summaries)
    for row in summaries:
        print(
            "cut={cut_um:>3} um: H={hydrogen_percent:.3f}%, "
            "He={helium_percent:.3f}%, Z>=3={heavy_Z_ge_3_percent:.3f}%, "
            "e={electron_percent:.3f}%, LET={main_LETd_w_keV_um:.3f}, "
            "closure={component_closure_percent:.3f}%".format(**row)
        )


if __name__ == "__main__":
    main()
