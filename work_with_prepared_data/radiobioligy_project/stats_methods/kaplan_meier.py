# файл kaplan_meier.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2, norm

from utils.plotting_helpers import PLOT_FONT_FAMILY
from data_processing.excel_data_processor import RatSurvivalEvent


@dataclass(frozen=True)
class KaplanMeierResult:
    """
    Ступенчатая оценка функции выживания S(t) методом Каплана-Майера.

    times[0]=0.0, survival[0]=1.0 — точка старта наблюдения. Каждый последующий элемент
    соответствует моменту, где произошла хотя бы одна смерть или цензурирование.

    Attributes:
        times:           моменты времени (дни)
        survival:        S(t) на каждом шаге
        n_at_risk:       число животных в риске непосредственно ПЕРЕД этим моментом
        n_events:        число смертей в этот момент
        n_censored:      число цензурирований в этот момент
        ci_lower/upper:  95% (или заданный confidence) доверительный интервал Гринвуда
        median_survival: день, когда S(t) впервые <= 0.5; None если не достигнут
    """
    times: Tuple[float, ...]
    survival: Tuple[float, ...]
    n_at_risk: Tuple[int, ...]
    n_events: Tuple[int, ...]
    n_censored: Tuple[int, ...]
    ci_lower: Tuple[float, ...]
    ci_upper: Tuple[float, ...]
    median_survival: Optional[float]


def _valid_observations(events: Sequence[RatSurvivalEvent]) -> List[Tuple[float, bool]]:
    return [(float(e.day), bool(e.event_observed)) for e in events if e.day is not None]


def kaplan_meier_estimate(events: Sequence[RatSurvivalEvent], *, confidence: float = 0.95) -> KaplanMeierResult:
    """
    Оценка Каплана-Майера по списку событий смерти/цензурирования одной группы животных.

    Цензурирование (event_observed=False) уменьшает число животных в риске, но не считается
    смертью — это и отличает корректную оценку выживаемости от наивного усреднения объёмов
    опухолей, где выбывшее животное просто перестаёт давать данные без учёта того, дожило ли
    оно до конца исследования или умерло раньше.

    Args:
        events:     список RatSurvivalEvent для одной группы (например, всех крыс из одного
                   файла/режима облучения).
        confidence: уровень доверительного интервала Гринвуда (по умолчанию 0.95).

    Returns:
        KaplanMeierResult. Если нет ни одного события с известным днём — вырожденный
        результат (S(t)=1.0 в единственной точке t=0).
    """
    observations = sorted(_valid_observations(events), key=lambda x: x[0])
    n_total = len(observations)
    if n_total == 0:
        return KaplanMeierResult((0.0,), (1.0,), (0,), (0,), (0,), (1.0,), (1.0,), None)

    unique_times = sorted(set(t for t, _ in observations))
    z = float(norm.ppf(0.5 + confidence / 2.0))

    times: List[float] = [0.0]
    survival: List[float] = [1.0]
    n_at_risk_list: List[int] = [n_total]
    n_events_list: List[int] = [0]
    n_censored_list: List[int] = [0]
    ci_lower: List[float] = [1.0]
    ci_upper: List[float] = [1.0]

    n_at_risk = n_total
    s = 1.0
    greenwood_var_sum = 0.0  # Σ d_i / (n_i (n_i - d_i))

    for t in unique_times:
        d_i = sum(1 for tt, observed in observations if tt == t and observed)
        c_i = sum(1 for tt, observed in observations if tt == t and not observed)

        if d_i > 0 and n_at_risk > 0:
            s *= (1.0 - d_i / n_at_risk)
            if n_at_risk > d_i:
                greenwood_var_sum += d_i / (n_at_risk * (n_at_risk - d_i))

        times.append(t)
        survival.append(s)
        n_at_risk_list.append(n_at_risk)
        n_events_list.append(d_i)
        n_censored_list.append(c_i)

        if 0.0 < s < 1.0 and greenwood_var_sum > 0.0:
            factor = z * np.sqrt(greenwood_var_sum)
            lower = float(s * np.exp(-factor))
            upper = float(s * np.exp(factor))
            ci_lower.append(max(0.0, min(1.0, lower)))
            ci_upper.append(max(0.0, min(1.0, upper)))
        else:
            ci_lower.append(s)
            ci_upper.append(s)

        n_at_risk -= (d_i + c_i)

    median_survival = None
    for t, s_val in zip(times, survival):
        if s_val <= 0.5:
            median_survival = t
            break

    return KaplanMeierResult(
        tuple(times), tuple(survival), tuple(n_at_risk_list), tuple(n_events_list),
        tuple(n_censored_list), tuple(ci_lower), tuple(ci_upper), median_survival,
    )


def log_rank_test(events_a: Sequence[RatSurvivalEvent], events_b: Sequence[RatSurvivalEvent]) -> Tuple[float, float]:
    """
    Лог-ранговый тест (Mantel-Haenszel) для сравнения времени до смерти между двумя группами.

    Args:
        events_a, events_b: списки RatSurvivalEvent для каждой из двух сравниваемых групп.

    Returns:
        (chi2_statistic, p_value) с 1 степенью свободы.
    """
    obs_a = _valid_observations(events_a)
    obs_b = _valid_observations(events_b)
    all_times = sorted(set(t for t, _ in obs_a) | set(t for t, _ in obs_b))

    observed_minus_expected = 0.0
    variance_sum = 0.0

    for t in all_times:
        at_risk_a = sum(1 for tt, _ in obs_a if tt >= t)
        at_risk_b = sum(1 for tt, _ in obs_b if tt >= t)
        n_at_t = at_risk_a + at_risk_b
        if n_at_t <= 1:
            continue

        d_a = sum(1 for tt, observed in obs_a if tt == t and observed)
        d_b = sum(1 for tt, observed in obs_b if tt == t and observed)
        d_total = d_a + d_b
        if d_total == 0:
            continue

        expected_a = d_total * (at_risk_a / n_at_t)
        observed_minus_expected += (d_a - expected_a)

        if n_at_t > 1:
            variance_sum += (
                d_total * (at_risk_a / n_at_t) * (at_risk_b / n_at_t) * (n_at_t - d_total)
            ) / (n_at_t - 1)

    if variance_sum <= 0.0:
        return 0.0, 1.0

    chi2_stat = (observed_minus_expected ** 2) / variance_sum
    p_value = float(chi2.sf(chi2_stat, df=1))
    return float(chi2_stat), p_value


def _step_value_at(km: KaplanMeierResult, t: float) -> float:
    value = km.survival[0]
    for tt, s in zip(km.times, km.survival):
        if tt <= t:
            value = s
        else:
            break
    return value


def plot_kaplan_meier(
        groups: Dict[str, Sequence[RatSurvivalEvent]],
        *,
        title: str = "Кривые выживаемости (Каплан-Майер)",
        x_label: str = "Время, сут.",
        y_label: str = "Доля выживших",
        show_censored_ticks: bool = True,
        show_ci: bool = False,
        figsize: Tuple[float, float] = (10, 6),
):
    """
    Рисует ступенчатые кривые Каплана-Майера для одной или нескольких групп животных.

    Args:
        groups:              {имя группы: список RatSurvivalEvent}
        show_censored_ticks: отмечать моменты цензурирования крестиком на кривой
        show_ci:             рисовать доверительный интервал Гринвуда (полупрозрачная область)

    Returns:
        (fig, ax)
    """
    with sns.axes_style("whitegrid", rc={'font.family': PLOT_FONT_FAMILY}):
        fig, ax = plt.subplots(figsize=figsize)
        for label, events in groups.items():
            km = kaplan_meier_estimate(events)
            line, = ax.step(km.times, km.survival, where='post', label=label, linewidth=2)

            if show_ci:
                ax.fill_between(km.times, km.ci_lower, km.ci_upper, step='post',
                                 alpha=0.15, color=line.get_color())

            if show_censored_ticks:
                observations = _valid_observations(events)
                censor_times = [t for t, observed in observations if not observed]
                censor_y = [_step_value_at(km, t) for t in censor_times]
                ax.plot(censor_times, censor_y, '+', markersize=10, markeredgewidth=1.5,
                         color=line.get_color())

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=14)
        fig.tight_layout()
        return fig, ax
