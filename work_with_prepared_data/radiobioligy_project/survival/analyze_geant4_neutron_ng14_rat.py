"""Summarise replicated NG-14 neutron calculations in the rat phantom.

The script analyses the reduced 0.4 x 0.4 x 0.2 mm phantom used by the
proton calculations.  The incident neutron field is described in
``neutron_ng14_measured_field.mac``.  Absolute Monte-Carlo dose values are
reported per simulated incident neutron.  Spatial dose indices are also
normalised to the pooled GTV mean because the retrospective archive does not
contain a complete series-specific fluence calibration for this CT geometry.

Fine-grid DVH values are deliberately retained as a Monte-Carlo sparsity
diagnostic.  The predeclared reporting grid for neutron spatial indices is
1.6 x 1.6 x 0.8 mm (4 x 4 x 4 fine voxels).
"""

from __future__ import annotations

import argparse
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

TASK_DIR = Path(r"C:\dev\dissertation\task4_5")
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from work_with_prepared_data.radiobioligy_project.survival.analyze_geant4_proton_100MeV_reduced_gtv import (  # noqa: E402
    DEFAULT_CT_DIR,
    DEFAULT_RS,
    NX,
    NY,
    NZ,
    PX_MM,
    PY_MM,
    PZ_MM,
    find_main_map,
    gtv_geometry,
    locally_aggregate_gtv_dose,
    parse_voxel_map,
)
from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (  # noqa: E402
    MatplotlibConfigurator,
)


DEFAULT_RUN_ROOT = TASK_DIR / "scoring_v2_neutron_ng14_rat"
DEFAULT_OUTPUT = (
    TASK_DIR
    / "outputs"
    / "geant4_livermore_20260724"
    / "neutron_ng14_rat"
)
REFERENCE_COMPONENTS_CSV = (
    TASK_DIR
    / "source_models"
    / "neutron"
    / "reference_component_contributions_14p5MeV.csv"
)
COMPONENTS = (
    "electron",
    "hydrogen",
    "helium",
    "ion_Z3_to_Z6",
    "ion_Z7_plus",
)
COMPONENT_LABELS = {
    "electron": r"$e^\pm$",
    "hydrogen": "H",
    "helium": "He",
    "ion_Z3_to_Z6": r"$Z=3$–$6$",
    "ion_Z7_plus": r"$Z\geq7$",
}
REPORT_FACTORS = (4, 4, 4)
REPORT_GRID_LABEL = "1.6 x 1.6 x 0.8 mm"


def number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def coefficient_of_variation(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2 or np.mean(array) == 0.0:
        return float("nan")
    return float(np.std(array, ddof=1) / np.mean(array))


def dose_at_volume(dose: np.ndarray, volume_percent: float) -> float:
    return float(np.percentile(dose, 100.0 - volume_percent))


def discover_runs(run_root: Path) -> list[Path]:
    runs: list[Path] = []
    for directory in sorted(run_root.iterdir()):
        if not directory.is_dir() or not directory.name.lower().startswith("seed"):
            continue
        try:
            find_main_map(directory)
        except RuntimeError:
            continue
        runs.append(directory)
    if not runs:
        raise RuntimeError(f"No complete seed runs found under {run_root}")
    return runs


def load_metadata(run_dir: Path) -> dict[str, object]:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        raise RuntimeError(f"Missing metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def find_component_map(run_dir: Path, component: str) -> Path:
    matches = list(run_dir.glob(f"vox_*_component_{component}_0"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {component} map in {run_dir}, found {len(matches)}"
        )
    return matches[0]


def repeated_block_means(
    dose: np.ndarray,
    gtv_ids: np.ndarray,
    factors: tuple[int, int, int] = REPORT_FACTORS,
) -> tuple[np.ndarray, int, float]:
    """Return one locally averaged value for every original GTV voxel."""
    factor_x, factor_y, factor_z = factors
    ix = gtv_ids % NX
    iy = (gtv_ids // NX) % NY
    iz = gtv_ids // (NX * NY)
    blocks_x = (NX + factor_x - 1) // factor_x
    blocks_y = (NY + factor_y - 1) // factor_y
    block_id = (
        ix // factor_x
        + blocks_x * (iy // factor_y)
        + blocks_x * blocks_y * (iz // factor_z)
    )
    _, inverse = np.unique(block_id, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=dose)
    means = sums / counts
    repeated = means[inverse]
    # DVH coverage is volume weighted, so report the fraction of original
    # fine GTV voxels represented by non-zero local blocks.
    nonzero_fraction = float(np.mean(repeated > 0.0))
    return repeated, int(means.size), nonzero_fraction


def analyse_seed(
    run_dir: Path,
    gtv_ids: np.ndarray,
    gtv_lookup: np.ndarray,
) -> tuple[dict[str, float | int | str], dict[str, np.ndarray]]:
    metadata = load_metadata(run_dir)
    histories = int(metadata["histories"])
    main = parse_voxel_map(
        find_main_map(run_dir),
        gtv_lookup,
        include_dose=True,
    )
    energy = np.asarray(main["gtv_energy_keV"], dtype=float)
    let_base = np.asarray(main["gtv_let_base_keV2_um"], dtype=float)
    dose = np.asarray(main["gtv_dose_Gy"], dtype=float)
    aggregate, aggregate_blocks = locally_aggregate_gtv_dose(
        dose,
        gtv_ids,
        REPORT_FACTORS,
    )
    mean_dose = float(np.mean(dose))

    arrays: dict[str, np.ndarray] = {
        "energy": energy,
        "let_base": let_base,
        "dose": dose,
    }
    component_metrics: dict[str, float] = {}
    for component in COMPONENTS:
        component_map = parse_voxel_map(
            find_component_map(run_dir, component),
            gtv_lookup,
            include_dose=False,
        )
        component_energy = np.asarray(
            component_map["gtv_energy_keV"],
            dtype=float,
        )
        component_let_base = np.asarray(
            component_map["gtv_let_base_keV2_um"],
            dtype=float,
        )
        arrays[f"{component}_energy"] = component_energy
        arrays[f"{component}_let_base"] = component_let_base
        component_metrics[f"{component}_energy_keV"] = float(
            np.sum(component_energy)
        )
        component_metrics[f"{component}_let_base_keV2_um"] = float(
            np.sum(component_let_base)
        )
        component_metrics[f"{component}_LETd_w_keV_um"] = finite_ratio(
            float(np.sum(component_let_base)),
            float(np.sum(component_energy)),
        )

    summary: dict[str, float | int | str] = {
        "run": run_dir.name,
        "histories": histories,
        "seed1": int(metadata["seed1"]),
        "seed2": int(metadata["seed2"]),
        "gtv_voxels": int(gtv_ids.size),
        "gtv_voxels_with_dose": int(np.count_nonzero(dose > 0.0)),
        "fine_nonzero_fraction": float(np.mean(dose > 0.0)),
        "report_blocks": aggregate_blocks,
        "report_nonzero_fraction": float(np.mean(aggregate > 0.0)),
        "gtv_depEnergy_keV": float(np.sum(energy)),
        "gtv_letBase_keV2_um": float(np.sum(let_base)),
        "Dmean_Gy_per_run": mean_dose,
        "Dmean_Gy_per_incident_neutron": mean_dose / histories,
        "incident_neutrons_per_Gy_GTV_mean": (
            1.0 / (mean_dose / histories) if mean_dose > 0.0 else float("nan")
        ),
        "LETd_w_all_keV_um": finite_ratio(
            float(np.sum(let_base)),
            float(np.sum(energy)),
        ),
        "fine_D90_over_Dmean": finite_ratio(
            dose_at_volume(dose, 90.0),
            mean_dose,
        ),
        "report_D50_over_Dmean": finite_ratio(
            dose_at_volume(aggregate, 50.0),
            mean_dose,
        ),
        "report_D90_over_Dmean": finite_ratio(
            dose_at_volume(aggregate, 90.0),
            mean_dose,
        ),
        "report_D95_over_Dmean": finite_ratio(
            dose_at_volume(aggregate, 95.0),
            mean_dose,
        ),
        **component_metrics,
    }
    return summary, arrays


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_reference_components() -> list[dict[str, float | str]]:
    if not REFERENCE_COMPONENTS_CSV.exists():
        raise RuntimeError(
            f"Missing reference component table: {REFERENCE_COMPONENTS_CSV}"
        )
    rows: list[dict[str, float | str]] = []
    with REFERENCE_COMPONENTS_CSV.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        for source in csv.DictReader(handle):
            row: dict[str, float | str] = dict(source)
            for field in (
                "protons_percent",
                "alpha_percent",
                "heavy_ions_percent",
                "electrons_percent",
                "absorbed_energy_MeV_per_g_per_source_particle",
            ):
                value = str(source.get(field, "")).strip()
                row[field] = float(value) if value else float("nan")
            rows.append(row)
    return rows


def collapsed_component_values(
    component_rows: list[dict[str, float | str]],
    field: str,
) -> dict[str, float]:
    by_component = {
        str(row["component"]): number(row[field]) for row in component_rows
    }
    return {
        "protons_percent": by_component["hydrogen"],
        "alpha_percent": by_component["helium"],
        "heavy_ions_percent": (
            by_component["ion_Z3_to_Z6"] + by_component["ion_Z7_plus"]
        ),
        "electrons_percent": by_component["electron"],
    }


def build_component_reference_figure(
    output: Path,
    component_rows: list[dict[str, float | str]],
    reference_rows: list[dict[str, float | str]],
) -> None:
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
        }
    )
    current = collapsed_component_values(
        component_rows,
        "dose_fraction_percent",
    )
    fields = (
        ("protons_percent", "Протоны / H"),
        ("alpha_percent", "Альфа-частицы / He"),
        ("heavy_ions_percent", r"Тяжёлые ионы / $Z\geq3$"),
        ("electrons_percent", r"Электроны / $e^\pm$"),
    )
    row_labels = [str(row["display_label"]) for row in reference_rows]
    row_labels.append("Настоящий расчёт: GTV крысы")
    y = np.arange(len(row_labels))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(17, 10.5),
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.25,
        right=0.985,
        top=0.86,
        bottom=0.13,
        hspace=0.34,
        wspace=0.08,
    )
    for panel_index, (ax, (field, title)) in enumerate(
        zip(axes.ravel(), fields)
    ):
        reference_values = np.array(
            [number(row[field]) for row in reference_rows],
            dtype=float,
        )
        reference_ids = [str(row["reference_id"]) for row in reference_rows]
        ordinary = np.array(
            [
                reference_id not in {"caswell", "qgsp_bic_hp"}
                for reference_id in reference_ids
            ],
            dtype=bool,
        )
        ax.scatter(
            reference_values[ordinary],
            y[:-1][ordinary],
            s=52,
            marker="o",
            facecolors="none",
            edgecolors="0.46",
            linewidths=1.2,
            label="остальные строки таблицы",
            zorder=3,
        )
        caswell_index = reference_ids.index("caswell")
        hp_index = reference_ids.index("qgsp_bic_hp")
        ax.scatter(
            reference_values[caswell_index],
            y[caswell_index],
            s=78,
            marker="D",
            color="#222222",
            label="Caswell",
            zorder=4,
        )
        ax.scatter(
            reference_values[hp_index],
            y[hp_index],
            s=88,
            marker="^",
            color="#7f7f7f",
            label="QGSP_BIC_HP",
            zorder=4,
        )
        current_value = current[field]
        ax.scatter(
            current_value,
            y[-1],
            s=105,
            marker="o",
            color="#c62828",
            label="настоящий расчёт",
            zorder=5,
        )
        all_values = np.append(reference_values, current_value)
        x_limit = max(all_values) * 1.18
        ax.set_xlim(0.0, x_limit)
        for value, y_value in zip(all_values, y):
            ax.text(
                value + x_limit * 0.012,
                y_value,
                f"{value:.1f}".replace(".", ","),
                va="center",
                fontsize=9,
            )
        ax.set_title(title)
        ax.set_xlabel("Доля энерговклада, %")
        ax.grid(axis="x", alpha=0.22)
        ax.set_ylim(len(row_labels) - 0.35, -0.65)
        if panel_index % 2 == 0:
            ax.set_yticks(y, row_labels)
        else:
            ax.tick_params(labelleft=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Компонентный состав поглощённой дозы при нейтронах 14,5–14,7 МэВ",
        fontsize=18,
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        (
            "Референсы перенесены из предоставленной таблицы; её первичный "
            "источник и геометрия требуют уточнения. Сопоставление не является "
            "геометрической валидацией модели крысы."
        ),
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.savefig(
        output / "neutron_component_reference_comparison.png",
        dpi=220,
    )
    fig.savefig(output / "neutron_component_reference_comparison.pdf")
    plt.close(fig)
    configurator.restore_original_styles()


def dvh_curve(dose_ratio: np.ndarray, points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    upper = max(1.25, float(np.percentile(dose_ratio, 99.8)) * 1.05)
    levels = np.linspace(0.0, upper, points)
    volume = np.array(
        [100.0 * np.mean(dose_ratio >= level) for level in levels],
        dtype=float,
    )
    return levels, volume


def build_figure(
    output: Path,
    pooled: dict[str, np.ndarray],
    gtv_ids: np.ndarray,
    component_rows: list[dict[str, float | str]],
    reference_rows: list[dict[str, float | str]],
    pooled_summary: dict[str, float | int | str],
) -> None:
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 17,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )

    dose = pooled["dose"]
    energy = pooled["energy"]
    let_base = pooled["let_base"]
    mean_dose = float(np.mean(dose))
    dose_ratio = dose / mean_dose
    report_dose_ratio, _, _ = repeated_block_means(dose_ratio, gtv_ids)

    ix = gtv_ids % NX
    iy = (gtv_ids // NX) % NY
    iz = gtv_ids // (NX * NY)
    z_counts = np.bincount(iz, minlength=NZ)
    z_slice = int(np.argmax(z_counts))
    smoothed_ratio, _, _ = repeated_block_means(dose_ratio, gtv_ids)
    dose_plane = np.full((NY, NX), np.nan, dtype=float)
    mask_plane = np.zeros((NY, NX), dtype=bool)
    on_slice = iz == z_slice
    dose_plane[iy[on_slice], ix[on_slice]] = smoothed_ratio[on_slice]
    mask_plane[iy[on_slice], ix[on_slice]] = True

    roi_counts = np.bincount(ix, minlength=NX)
    depth_dose = np.bincount(ix, weights=dose, minlength=NX)
    depth_energy = np.bincount(ix, weights=energy, minlength=NX)
    depth_let_base = np.bincount(ix, weights=let_base, minlength=NX)
    depth_mean_dose = np.divide(
        depth_dose,
        roi_counts,
        out=np.full(NX, np.nan),
        where=roi_counts > 0,
    )
    depth_let = np.divide(
        depth_let_base,
        depth_energy,
        out=np.full(NX, np.nan),
        where=depth_energy > 0,
    )
    hydrogen_energy = pooled["hydrogen_energy"]
    hydrogen_let_base = pooled["hydrogen_let_base"]
    depth_h_energy = np.bincount(ix, weights=hydrogen_energy, minlength=NX)
    depth_h_let_base = np.bincount(ix, weights=hydrogen_let_base, minlength=NX)
    depth_h_let = np.divide(
        depth_h_let_base,
        depth_h_energy,
        out=np.full(NX, np.nan),
        where=depth_h_energy > 0,
    )
    depth_kernel = np.ones(3, dtype=float)
    smooth_roi_counts = np.convolve(roi_counts, depth_kernel, mode="same")
    smooth_depth_dose = np.divide(
        np.convolve(depth_dose, depth_kernel, mode="same"),
        smooth_roi_counts,
        out=np.full(NX, np.nan),
        where=smooth_roi_counts > 0,
    )
    smooth_depth_let = np.divide(
        np.convolve(depth_let_base, depth_kernel, mode="same"),
        np.convolve(depth_energy, depth_kernel, mode="same"),
        out=np.full(NX, np.nan),
        where=np.convolve(depth_energy, depth_kernel, mode="same") > 0,
    )
    smooth_depth_h_let = np.divide(
        np.convolve(depth_h_let_base, depth_kernel, mode="same"),
        np.convolve(depth_h_energy, depth_kernel, mode="same"),
        out=np.full(NX, np.nan),
        where=np.convolve(depth_h_energy, depth_kernel, mode="same") > 0,
    )
    depth_mm = (np.arange(NX) + 0.5) * PX_MM

    fig, axes = plt.subplots(2, 2, figsize=(17, 13), constrained_layout=True)

    ax = axes[0, 0]
    image = ax.imshow(
        dose_plane,
        origin="lower",
        extent=(0.0, NX * PX_MM, 0.0, NY * PY_MM),
        aspect="equal",
        cmap="magma",
        vmin=0.0,
        vmax=float(np.nanpercentile(dose_plane, 98.0)),
    )
    ax.contour(
        mask_plane.astype(float),
        levels=[0.5],
        colors=["cyan"],
        linewidths=1.5,
        origin="lower",
        extent=(0.0, NX * PX_MM, 0.0, NY * PY_MM),
    )
    ax.annotate(
        "пучок +x",
        xy=(13.0, 47.0),
        xytext=(2.0, 47.0),
        arrowprops={"arrowstyle": "->", "color": "white", "lw": 2.0},
        color="white",
        va="center",
    )
    ax.set_title(
        "Доза в GTV, слой максимальной площади\n"
        f"сетка {REPORT_GRID_LABEL}, $D/D_{{mean}}$"
    )
    ax.set_xlabel("x — глубина в фантоме, мм")
    ax.set_ylabel("y, мм")
    fig.colorbar(image, ax=ax, label=r"$D/D_{mean}$", shrink=0.86)

    ax = axes[0, 1]
    valid = roi_counts > 0
    ax.plot(
        depth_mm[valid],
        smooth_depth_dose[valid] / mean_dose,
        color="#d24b40",
        lw=2.4,
        label=r"$D/D_{mean}$",
    )
    ax.axhline(1.0, color="0.45", lw=1.0, ls="--")
    ax.set_xlabel("x — глубина в фантоме, мм")
    ax.set_ylabel(r"$D/D_{mean}$", color="#d24b40")
    ax.tick_params(axis="y", labelcolor="#d24b40")
    ax.set_title("Глубинный профиль внутри GTV")
    ax.grid(alpha=0.22)
    let_axis = ax.twinx()
    let_axis.plot(
        depth_mm[valid],
        smooth_depth_let[valid],
        color="#512a8a",
        lw=1.8,
        label=r"$LET_{D,w}$, все",
    )
    let_axis.plot(
        depth_mm[valid],
        smooth_depth_h_let[valid],
        color="#198f70",
        lw=1.8,
        label=r"$LET_{D,w}$, H",
    )
    let_axis.set_ylabel(r"$LET_{D,w}$, кэВ/мкм")
    lines = ax.lines[:1] + let_axis.lines
    ax.legend(lines, [line.get_label() for line in lines], loc="upper right")

    ax = axes[1, 0]
    fine_x, fine_y = dvh_curve(dose_ratio)
    report_x, report_y = dvh_curve(report_dose_ratio)
    ax.plot(
        fine_x,
        fine_y,
        color="0.55",
        lw=1.7,
        ls="--",
        label="сетка 0,4×0,4×0,2 мм",
    )
    ax.plot(
        report_x,
        report_y,
        color="#c62828",
        lw=2.7,
        label="усреднение 1,6×1,6×0,8 мм",
    )
    ax.axvline(1.0, color="0.35", lw=1.0, ls=":")
    report_upper = max(2.0, float(np.percentile(report_dose_ratio, 99.5)) * 1.08)
    ax.set_xlim(0.0, min(10.0, report_upper))
    ax.set_ylim(0.0, 101.0)
    ax.set_xlabel(r"$D/D_{mean}$")
    ax.set_ylabel("Объём GTV, %")
    ax.set_title("Кумулятивная DVH (статистика не сошлась)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    by_component = {
        str(row["component"]): row for row in component_rows
    }
    labels = [
        "H / протоны",
        "He / альфа",
        r"$Z=3$–$6$",
        r"$Z\geq7$",
        r"$\Sigma Z\geq3$ (сумма)",
        r"$e^\pm$",
    ]
    dose_fractions = np.array(
        [
            number(by_component["hydrogen"]["dose_fraction_percent"]),
            number(by_component["helium"]["dose_fraction_percent"]),
            number(by_component["ion_Z3_to_Z6"]["dose_fraction_percent"]),
            number(by_component["ion_Z7_plus"]["dose_fraction_percent"]),
            (
                number(
                    by_component["ion_Z3_to_Z6"]["dose_fraction_percent"]
                )
                + number(
                    by_component["ion_Z7_plus"]["dose_fraction_percent"]
                )
            ),
            number(by_component["electron"]["dose_fraction_percent"]),
        ],
        dtype=float,
    )
    numerator_fractions = np.array(
        [
            number(
                by_component["hydrogen"][
                    "LET_numerator_fraction_percent"
                ]
            ),
            number(
                by_component["helium"]["LET_numerator_fraction_percent"]
            ),
            number(
                by_component["ion_Z3_to_Z6"][
                    "LET_numerator_fraction_percent"
                ]
            ),
            number(
                by_component["ion_Z7_plus"][
                    "LET_numerator_fraction_percent"
                ]
            ),
            (
                number(
                    by_component["ion_Z3_to_Z6"][
                        "LET_numerator_fraction_percent"
                    ]
                )
                + number(
                    by_component["ion_Z7_plus"][
                        "LET_numerator_fraction_percent"
                    ]
                )
            ),
            number(
                by_component["electron"][
                    "LET_numerator_fraction_percent"
                ]
            ),
        ],
        dtype=float,
    )
    y = np.arange(len(labels))
    ax.axhspan(
        3.55,
        4.45,
        color="0.94",
        zorder=0,
    )
    ax.barh(
        y,
        dose_fractions,
        0.46,
        color="#4c78a8",
        alpha=0.88,
        label="настоящий расчёт: энерговклад",
        zorder=2,
    )
    ax.scatter(
        numerator_fractions,
        y,
        s=74,
        marker="s",
        color="#f58518",
        edgecolors="white",
        linewidths=0.8,
        label="настоящий расчёт: LET-числитель",
        zorder=5,
    )
    fields = (
        "protons_percent",
        "alpha_percent",
        "heavy_ions_percent",
        "electrons_percent",
    )
    reference_y = np.array([0.0, 1.0, 4.0, 5.0])
    reference_matrix = np.array(
        [
            [number(row[field]) for field in fields]
            for row in reference_rows
        ],
        dtype=float,
    )
    reference_ids = [str(row["reference_id"]) for row in reference_rows]
    ordinary_indices = [
        index
        for index, reference_id in enumerate(reference_ids)
        if reference_id not in {"caswell", "qgsp_bic_hp"}
    ]
    offsets = np.linspace(-0.17, 0.17, max(1, len(ordinary_indices)))
    for ordinary_order, (offset, reference_index) in enumerate(
        zip(offsets, ordinary_indices)
    ):
        ax.scatter(
            reference_matrix[reference_index],
            reference_y + offset,
            s=31,
            marker="o",
            facecolors="none",
            edgecolors="0.48",
            linewidths=0.9,
            label=(
                "остальные строки таблицы"
                if ordinary_order == 0
                else None
            ),
            zorder=4,
        )
    caswell_index = reference_ids.index("caswell")
    hp_index = reference_ids.index("qgsp_bic_hp")
    ax.scatter(
        reference_matrix[caswell_index],
        reference_y - 0.06,
        s=63,
        marker="D",
        color="#202020",
        label="Caswell",
        zorder=6,
    )
    ax.scatter(
        reference_matrix[hp_index],
        reference_y + 0.06,
        s=72,
        marker="^",
        color="#7f7f7f",
        label="QGSP_BIC_HP",
        zorder=6,
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("%")
    ax.set_title("Компоненты дозы и референсные точки")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False,
    )

    histories_label = f"{int(pooled_summary['total_histories']):,}".replace(
        ",",
        " ",
    )
    fig.suptitle(
        f"НГ-14, 14,7 МэВ — {histories_label} нейтронов; "
        "QGSP_BIC_AllHP + Livermore",
        fontsize=18,
    )
    fig.savefig(output / "neutron_ng14_rat_summary.png", dpi=220)
    fig.savefig(output / "neutron_ng14_rat_summary.pdf")
    plt.close(fig)
    configurator.restore_original_styles()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ct-dir", type=Path, default=Path(DEFAULT_CT_DIR))
    parser.add_argument("--rs", type=Path, default=Path(DEFAULT_RS))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    _, gtv_ids, gtv_lookup, _ = gtv_geometry(args.ct_dir, args.rs)
    run_dirs = discover_runs(args.run_root)

    seed_rows: list[dict[str, float | int | str]] = []
    seed_arrays: list[dict[str, np.ndarray]] = []
    for run_dir in run_dirs:
        summary, arrays = analyse_seed(run_dir, gtv_ids, gtv_lookup)
        seed_rows.append(summary)
        seed_arrays.append(arrays)
        print(
            f"{run_dir.name}: n={summary['histories']}, "
            f"LET(all)={number(summary['LETd_w_all_keV_um']):.3f}, "
            f"LET(H)={number(summary['hydrogen_LETd_w_keV_um']):.3f}, "
            f"D90/Dmean={number(summary['report_D90_over_Dmean']):.3f}"
        )

    write_csv(args.output / "neutron_ng14_seed_results.csv", seed_rows)

    pooled: dict[str, np.ndarray] = {}
    for key in seed_arrays[0]:
        pooled[key] = np.sum(
            np.stack([arrays[key] for arrays in seed_arrays], axis=0),
            axis=0,
        )
    total_histories = int(sum(int(row["histories"]) for row in seed_rows))
    dose = pooled["dose"]
    energy = pooled["energy"]
    let_base = pooled["let_base"]
    mean_dose = float(np.mean(dose))
    report_dose, report_blocks, report_nonzero = repeated_block_means(
        dose,
        gtv_ids,
    )

    total_energy = float(np.sum(energy))
    total_let_base = float(np.sum(let_base))
    source_radius_cm = 1.73
    source_area_cm2 = float(np.pi * source_radius_cm**2)
    incident_neutrons_per_gy = 1.0 / (mean_dose / total_histories)
    component_rows: list[dict[str, float | str]] = []
    for component in COMPONENTS:
        component_energy = float(np.sum(pooled[f"{component}_energy"]))
        component_let_base = float(np.sum(pooled[f"{component}_let_base"]))
        component_rows.append(
            {
                "component": component,
                "label": COMPONENT_LABELS[component],
                "depEnergy_keV": component_energy,
                "letBase_keV2_um": component_let_base,
                "LETd_w_keV_um": finite_ratio(
                    component_let_base,
                    component_energy,
                ),
                "dose_fraction_percent": (
                    100.0 * finite_ratio(component_energy, total_energy)
                ),
                "LET_numerator_fraction_percent": (
                    100.0 * finite_ratio(component_let_base, total_let_base)
                ),
                "contribution_to_total_LETd_w_keV_um": finite_ratio(
                    component_let_base,
                    total_energy,
                ),
            }
        )
    write_csv(
        args.output / "neutron_ng14_component_summary.csv",
        component_rows,
    )
    reference_rows = load_reference_components()
    current_dose_components = collapsed_component_values(
        component_rows,
        "dose_fraction_percent",
    )
    reference_comparison_rows: list[dict[str, object]] = [
        dict(row) for row in reference_rows
    ]
    reference_comparison_rows.append(
        {
            "reference_id": "current_rat_gtv",
            "display_label": "Present calculation: rat GTV",
            **current_dose_components,
            "absorbed_energy_MeV_per_g_per_source_particle": "",
            "source_status": (
                "Current QGSP_BIC_AllHP + Livermore calculation in the rat GTV; "
                "absolute energy-per-mass value is not compared because the "
                "reference geometry is not documented"
            ),
        }
    )
    write_csv(
        args.output / "neutron_component_reference_comparison.csv",
        reference_comparison_rows,
    )

    pooled_summary: dict[str, float | int | str] = {
        "seed_runs": len(seed_rows),
        "total_histories": total_histories,
        "gtv_voxels": int(gtv_ids.size),
        "gtv_volume_cm3": float(gtv_ids.size * PX_MM * PY_MM * PZ_MM / 1000.0),
        "fine_grid_mm": f"{PX_MM} x {PY_MM} x {PZ_MM}",
        "report_grid_mm": REPORT_GRID_LABEL,
        "report_blocks": report_blocks,
        "fine_nonzero_fraction": float(np.mean(dose > 0.0)),
        "report_nonzero_fraction": report_nonzero,
        "Dmean_Gy_per_pooled_run": mean_dose,
        "Dmean_Gy_per_incident_neutron": mean_dose / total_histories,
        "source_field_area_cm2": source_area_cm2,
        "incident_neutrons_per_Gy_GTV_mean": incident_neutrons_per_gy,
        "incident_fluence_per_Gy_GTV_mean_n_cm2": (
            incident_neutrons_per_gy / source_area_cm2
        ),
        "LETd_w_all_keV_um": finite_ratio(total_let_base, total_energy),
        "LETd_w_hydrogen_keV_um": finite_ratio(
            float(np.sum(pooled["hydrogen_let_base"])),
            float(np.sum(pooled["hydrogen_energy"])),
        ),
        "LETd_w_hydrogen_plus_helium_keV_um": finite_ratio(
            float(
                np.sum(pooled["hydrogen_let_base"])
                + np.sum(pooled["helium_let_base"])
            ),
            float(
                np.sum(pooled["hydrogen_energy"])
                + np.sum(pooled["helium_energy"])
            ),
        ),
        "hydrogen_dose_fraction_percent": (
            100.0
            * finite_ratio(
                float(np.sum(pooled["hydrogen_energy"])),
                total_energy,
            )
        ),
        "report_D2_over_Dmean": finite_ratio(
            dose_at_volume(report_dose, 2.0),
            mean_dose,
        ),
        "report_D50_over_Dmean": finite_ratio(
            dose_at_volume(report_dose, 50.0),
            mean_dose,
        ),
        "report_D90_over_Dmean": finite_ratio(
            dose_at_volume(report_dose, 90.0),
            mean_dose,
        ),
        "report_D95_over_Dmean": finite_ratio(
            dose_at_volume(report_dose, 95.0),
            mean_dose,
        ),
        "seed_CV_Dmean_per_primary": coefficient_of_variation(
            [number(row["Dmean_Gy_per_incident_neutron"]) for row in seed_rows]
        ),
        "seed_CV_LETd_w_all": coefficient_of_variation(
            [number(row["LETd_w_all_keV_um"]) for row in seed_rows]
        ),
        "seed_CV_LETd_w_hydrogen": coefficient_of_variation(
            [number(row["hydrogen_LETd_w_keV_um"]) for row in seed_rows]
        ),
        "seed_CV_hydrogen_dose_fraction": coefficient_of_variation(
            [
                100.0
                * finite_ratio(
                    number(row["hydrogen_energy_keV"]),
                    number(row["gtv_depEnergy_keV"]),
                )
                for row in seed_rows
            ]
        ),
        "seed_CV_report_D90_over_Dmean": coefficient_of_variation(
            [number(row["report_D90_over_Dmean"]) for row in seed_rows]
        ),
        "spatial_DVH_converged": (
            "yes"
            if (
                min(number(row["report_nonzero_fraction"]) for row in seed_rows)
                >= 0.95
                and min(
                    number(row["report_D90_over_Dmean"]) for row in seed_rows
                )
                > 0.0
                and number(
                    coefficient_of_variation(
                        [
                            number(row["report_D90_over_Dmean"])
                            for row in seed_rows
                        ]
                    )
                )
                <= 0.10
            )
            else "no"
        ),
        "physics_list": "QGSP_BIC_AllHP + G4EmLivermorePhysics",
        "source": "14.7 MeV Gaussian sigma 0.15 MeV; equivalent 17.3 mm radius incident field",
        "source_status": "representative measured-field model, not a complete collimator reconstruction",
    }
    write_csv(
        args.output / "neutron_ng14_pooled_summary.csv",
        [pooled_summary],
    )

    depth_ix = gtv_ids % NX
    depth_rows: list[dict[str, float | int]] = []
    for ix_value in range(NX):
        selected = depth_ix == ix_value
        if not np.any(selected):
            continue
        depth_energy = float(np.sum(energy[selected]))
        depth_let_base = float(np.sum(let_base[selected]))
        depth_h_energy = float(np.sum(pooled["hydrogen_energy"][selected]))
        depth_h_let_base = float(
            np.sum(pooled["hydrogen_let_base"][selected])
        )
        depth_rows.append(
            {
                "x_index": ix_value,
                "depth_from_phantom_entrance_mm": (ix_value + 0.5) * PX_MM,
                "gtv_voxels": int(np.count_nonzero(selected)),
                "dose_sum_Gy": float(np.sum(dose[selected])),
                "mean_dose_Gy": float(np.mean(dose[selected])),
                "mean_dose_over_GTV_Dmean": float(
                    np.mean(dose[selected]) / mean_dose
                ),
                "depEnergy_keV": depth_energy,
                "LETd_w_all_keV_um": finite_ratio(
                    depth_let_base,
                    depth_energy,
                ),
                "LETd_w_hydrogen_keV_um": finite_ratio(
                    depth_h_let_base,
                    depth_h_energy,
                ),
            }
        )
    write_csv(args.output / "neutron_ng14_gtv_depth_profile.csv", depth_rows)

    fine_ratio = dose / mean_dose
    report_ratio = report_dose / mean_dose
    fine_x, fine_y = dvh_curve(fine_ratio)
    report_x, report_y = dvh_curve(report_ratio)
    with (args.output / "neutron_ng14_normalised_dvh.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "fine_D_over_Dmean",
                "fine_volume_percent",
                "report_D_over_Dmean",
                "report_volume_percent",
            ]
        )
        for values in zip(fine_x, fine_y, report_x, report_y):
            writer.writerow(values)

    build_figure(
        args.output,
        pooled,
        gtv_ids,
        component_rows,
        reference_rows,
        pooled_summary,
    )
    build_component_reference_figure(
        args.output,
        component_rows,
        reference_rows,
    )

    report = [
        "# Нейтроны НГ-14 в геометрии крысы",
        "",
        "Статус: реплицированный расчёт представительного входного поля.",
        "",
        "## Постановка",
        "",
        "- источник: 14,7 МэВ, Gaussian sigma 0,15 МэВ;",
        "- эквивалентный радиус измеренного плато: 17,3 мм;",
        "- транспорт: QGSP_BIC_AllHP + G4EmLivermorePhysics;",
        f"- число независимых seed: {len(seed_rows)};",
        f"- всего первичных нейтронов: {total_histories:,};".replace(",", " "),
        f"- GTVp: {pooled_summary['gtv_volume_cm3']:.3f} см3;",
        f"- отчётная пространственная сетка: {REPORT_GRID_LABEL}.",
        "",
        "## Основные оценки",
        "",
        (
            f"- LETd,w всех заряженных вторичных частиц: "
            f"{number(pooled_summary['LETd_w_all_keV_um']):.3f} кэВ/мкм;"
        ),
        (
            f"- LETd,w водородной компоненты: "
            f"{number(pooled_summary['LETd_w_hydrogen_keV_um']):.3f} кэВ/мкм;"
        ),
        (
            f"- LETd,w H+He: "
            f"{number(pooled_summary['LETd_w_hydrogen_plus_helium_keV_um']):.3f} "
            "кэВ/мкм;"
        ),
        (
            f"- доля водородной компоненты в энерговкладе: "
            f"{number(pooled_summary['hydrogen_dose_fraction_percent']):.2f}%;"
        ),
        (
            f"- D90/Dmean на отчётной сетке: "
            f"{number(pooled_summary['report_D90_over_Dmean']):.3f};"
        ),
        (
            f"- D95/Dmean на отчётной сетке: "
            f"{number(pooled_summary['report_D95_over_Dmean']):.3f};"
        ),
        (
            f"- incident neutrons per 1 Gy of mean GTV dose: "
            f"{number(pooled_summary['incident_neutrons_per_Gy_GTV_mean']):.6g}."
        ),
        (
            f"- соответствующий флюенс во входной плоскости: "
            f"{number(pooled_summary['incident_fluence_per_Gy_GTV_mean_n_cm2']):.6g} "
            "нейтронов/см2 на 1 Гр средней дозы GTV."
        ),
        "",
        "## Сходимость между seed",
        "",
        (
            f"- CV Dmean на первичный нейтрон: "
            f"{100.0 * number(pooled_summary['seed_CV_Dmean_per_primary']):.2f}%;"
        ),
        (
            f"- CV LETd,w(all): "
            f"{100.0 * number(pooled_summary['seed_CV_LETd_w_all']):.2f}%;"
        ),
        (
            f"- CV LETd,w(H): "
            f"{100.0 * number(pooled_summary['seed_CV_LETd_w_hydrogen']):.2f}%;"
        ),
        (
            f"- CV доли водородной компоненты: "
            f"{100.0 * number(pooled_summary['seed_CV_hydrogen_dose_fraction']):.2f}%;"
        ),
        (
            f"- CV D90/Dmean: "
            f"{100.0 * number(pooled_summary['seed_CV_report_D90_over_Dmean']):.2f}%."
        ),
        (
            f"- формальный статус сходимости пространственной DVH: "
            f"{pooled_summary['spatial_DVH_converged']}."
        ),
        "",
        (
            "Если статус DVH равен `no`, объединённые D90/D95 являются только "
            "диагностикой накопленной статистики и не переносятся в итоговую "
            "таблицу физических параметров. Для них требуется увеличение "
            "числа историй либо низкодисперсионный track-length/kerma estimator."
        ),
        "",
        "## Внешняя физическая согласованность",
        "",
        (
            "Полученная доля водородной компоненты сопоставима с опубликованной "
            "оценкой порядка 70% поглощённой дозы от протонов отдачи при "
            "14–15 МэВ нейтронах. Это поддерживает компонентную декомпозицию, "
            "но не является валидацией абсолютной дозы конкретной архивной серии "
            "(Isaeva et al., DOI 10.1093/rpd/nct247)."
        ),
        "",
        (
            "Дополнительно выполнено графическое сопоставление с предоставленной "
            "таблицей компонентных вкладов для нейтронов 14,5 МэВ. Настоящий "
            "расчёт дал H 76,82%, He 11,68%, тяжёлые ионы Z>=3 10,62% и "
            "электроны 0,88%. Значения близки к строке Caswell "
            "(72,9%, 12,4%, 10,9% и 1,7%), особенно по H, He и тяжёлым ионам. "
            "Однако первичный источник, геометрия и нормировка предоставленной "
            "таблицы пока не установлены, поэтому это сопоставление обозначено "
            "как референсное, а не как независимая валидация."
        ),
        "",
        "## Ограничение",
        "",
        (
            "В основной постановке использовано эквивалентное измеренное поле у "
            "объекта. Стальной коллиматор высотой 298 мм и формируемая им "
            "низкоэнергетическая рассеянная компонента явно не моделировались. "
            "Поэтому результат характеризует репрезентативное поле НГ-14, но "
            "ещё не является полной реконструкцией конкретной архивной серии."
        ),
        "",
        (
            "Литературная геометрическая опора: Мещанинов и соавт., 2024, "
            "DOI 10.31857/S0032816224040057."
        ),
    ]
    (args.output / "NEUTRON_NG14_RESULTS.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
