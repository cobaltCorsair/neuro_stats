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


@dataclass(frozen=True)
class HazardRatioResult:
    """
    Приближённая оценка отношения рисков (Mantel-Haenszel) группы A относительно
    группы B, выведенная из тех же слагаемых, что и лог-ранговый тест.

    HR > 1 означает более высокий риск смерти (худшую выживаемость) в группе A.
    """
    hazard_ratio: float
    ci_lower: float
    ci_upper: float


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


def median_survival_ci(km: KaplanMeierResult) -> Tuple[Optional[float], Optional[float]]:
    """
    Доверительный интервал медианы выживаемости (метод Брукмейера-Кроули, см. Klein &
    Moeschberger, разд. 4.4): инвертирует доверительные полосы S(t) — границы медианы там,
    где нижняя/верхняя полоса пересекает уровень 0.5, а не точечная оценка S(t).

    L = inf{t: НИЖНЯЯ граница ДИ для S(t) ≤ 0.5} — медиана не может быть раньше: даже в
        пессимистичном сценарии (нижняя граница) S(t) только сейчас опустилась до 0.5.
    U = inf{t: ВЕРХНЯЯ граница ДИ для S(t) ≤ 0.5} — медиана не может быть позже: даже в
        оптимистичном сценарии (верхняя граница) S(t) уже опустилась до 0.5.

    Returns:
        (нижняя граница, верхняя граница) в днях. None на любой стороне, если
        соответствующая полоса не достигает 0.5 в пределах наблюдения (граница не достигнута).
    """
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    for t, lower in zip(km.times, km.ci_lower):
        if lower <= 0.5:
            lower_bound = t
            break
    for t, upper in zip(km.times, km.ci_upper):
        if upper <= 0.5:
            upper_bound = t
            break
    return lower_bound, upper_bound


def restricted_mean_survival_time(km: KaplanMeierResult, tau: Optional[float] = None) -> float:
    """
    Restricted mean survival time — площадь под ступенчатой кривой S(t) от 0 до tau
    (по умолчанию — последний наблюдённый момент). Содержательная сводная характеристика
    даже когда медиана не достигнута (S(t) не опускается до 0.5 за время наблюдения).
    """
    if len(km.times) < 2:
        return 0.0
    cutoff = km.times[-1] if tau is None else float(tau)
    area = 0.0
    for i in range(len(km.times) - 1):
        t0, t1 = km.times[i], km.times[i + 1]
        if t0 >= cutoff:
            break
        segment_end = min(t1, cutoff)
        area += km.survival[i] * (segment_end - t0)
    return area


def _log_rank_components(
        events_a: Sequence[RatSurvivalEvent], events_b: Sequence[RatSurvivalEvent]
) -> Tuple[float, float]:
    """(observed_minus_expected, variance_sum) для группы A — общие слагаемые
    лог-рангового теста и приближённой оценки hazard ratio (Mantel-Haenszel)."""
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

    return observed_minus_expected, variance_sum


def log_rank_test(events_a: Sequence[RatSurvivalEvent], events_b: Sequence[RatSurvivalEvent]) -> Tuple[float, float]:
    """
    Лог-ранговый тест (Mantel-Haenszel) для сравнения времени до смерти между двумя группами.

    Args:
        events_a, events_b: списки RatSurvivalEvent для каждой из двух сравниваемых групп.

    Returns:
        (chi2_statistic, p_value) с 1 степенью свободы.
    """
    observed_minus_expected, variance_sum = _log_rank_components(events_a, events_b)
    if variance_sum <= 0.0:
        return 0.0, 1.0

    chi2_stat = (observed_minus_expected ** 2) / variance_sum
    p_value = float(chi2.sf(chi2_stat, df=1))
    return float(chi2_stat), p_value


def hazard_ratio_log_rank(
        events_a: Sequence[RatSurvivalEvent],
        events_b: Sequence[RatSurvivalEvent],
        *,
        confidence: float = 0.95,
) -> Optional[HazardRatioResult]:
    """
    Приближённая оценка hazard ratio (Mantel-Haenszel) группы A относительно группы B,
    из тех же O-E/V, что и лог-ранговый тест: ln(HR) = (O-E)/V, SE(ln HR) = sqrt(1/V).
    Это стандартное упрощение, принятое наравне с лог-рангом (не полноценная Cox-регрессия).

    Returns:
        HazardRatioResult, либо None если дисперсия нулевая (нет общих интервалов риска
        с хотя бы одним событием — оценка не определена).
    """
    observed_minus_expected, variance_sum = _log_rank_components(events_a, events_b)
    if variance_sum <= 0.0:
        return None

    z = float(norm.ppf(0.5 + confidence / 2.0))
    log_hr = observed_minus_expected / variance_sum
    se_log_hr = 1.0 / np.sqrt(variance_sum)
    hr = float(np.exp(log_hr))
    ci_lower = float(np.exp(log_hr - z * se_log_hr))
    ci_upper = float(np.exp(log_hr + z * se_log_hr))
    return HazardRatioResult(hr, ci_lower, ci_upper)


def _n_at_risk_at_time(events: Sequence[RatSurvivalEvent], t: float) -> int:
    """Число животных, для которых день события/цензуры >= t (т.е. ещё под наблюдением в момент t)."""
    return sum(1 for e in events if e.day is not None and e.day >= t)


def risk_table_time_points(groups: Dict[str, Sequence[RatSurvivalEvent]], n_points: int = 7) -> List[float]:
    """
    n_points равномерно распределённых моментов времени (целые сутки, без повторов)
    в пределах диапазона наблюдения — для отображения таблицы «число в риске».
    """
    max_time = 0.0
    for events in groups.values():
        for e in events:
            if e.day is not None:
                max_time = max(max_time, e.day)
    if max_time <= 0:
        return [0.0]

    points: List[float] = []
    seen = set()
    for raw in np.linspace(0, max_time, num=max(2, n_points)):
        rounded = round(raw)
        if rounded not in seen:
            seen.add(rounded)
            points.append(float(rounded))
    return points


def n_at_risk_table(
        groups: Dict[str, Sequence[RatSurvivalEvent]], time_points: Sequence[float]
) -> Dict[str, List[int]]:
    """{имя группы: [число в риске на каждый момент из time_points]}."""
    return {name: [_n_at_risk_at_time(events, t) for t in time_points] for name, events in groups.items()}


def max_observed_day(events: Sequence[RatSurvivalEvent]) -> Optional[float]:
    """Последний день наблюдения в группе (смерть/цензура), None если нет ни одного события."""
    days = [e.day for e in events if e.day is not None]
    return max(days) if days else None


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

    Таблицу «число в риске» график не включает — см. risk_table_time_points()/
    n_at_risk_table() и отдельный QTableWidget в KaplanMeierWindow: встраивание этой
    таблицы прямо в matplotlib-рисунок плохо масштабировалось при показе в Qt-окне.

    Args:
        groups:              {имя группы: список RatSurvivalEvent}
        show_censored_ticks: отмечать моменты цензурирования крестиком на кривой
        show_ci:             рисовать доверительный интервал Гринвуда (полупрозрачная область)

    Returns:
        (fig, ax)
    """
    with sns.axes_style("whitegrid", rc={'font.family': PLOT_FONT_FAMILY}):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

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

        return fig, ax
