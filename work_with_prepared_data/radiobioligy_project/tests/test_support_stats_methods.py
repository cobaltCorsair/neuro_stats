import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
for path in (str(WORKSPACE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from stats_methods.support_stats_methods import SupportingFunctions as SF


class TestCalculateStdDev(unittest.TestCase):
    def test_uses_valid_count_not_total_length(self):
        # 3 валидных значения (10,12,14) + 2 NaN (выбывшие животные)
        values = [10, 12, 14, float("nan"), float("nan")]
        std = SF.calculate_std_dev(values, mean_value=12.0)
        self.assertAlmostEqual(std, math.sqrt(8 / 2), places=9)  # n=3 -> /(3-1)

    def test_no_nan_matches_naive_formula(self):
        values = [10, 12, 14]
        std = SF.calculate_std_dev(values, mean_value=12.0)
        self.assertAlmostEqual(std, math.sqrt(8 / 2), places=9)

    def test_fewer_than_two_valid_returns_nan(self):
        self.assertTrue(math.isnan(SF.calculate_std_dev([5.0, float("nan")], 5.0)))
        self.assertTrue(math.isnan(SF.calculate_std_dev([], 0.0)))


class TestCalculateErrorMargin(unittest.TestCase):
    def test_correct_n_gives_wider_interval_than_buggy_total_n(self):
        std = math.sqrt(8 / 2)  # = 2.0, n_valid=3
        sem_correct = SF.calculate_error_margin(std, 3)
        sem_with_old_buggy_total_n = SF.calculate_error_margin(std, 5)
        self.assertAlmostEqual(sem_correct, 2.0 / math.sqrt(3), places=9)
        self.assertGreater(sem_correct, sem_with_old_buggy_total_n)

    def test_zero_n_returns_nan_instead_of_crashing(self):
        self.assertTrue(math.isnan(SF.calculate_error_margin(1.0, 0)))


class TestCountAtRisk(unittest.TestCase):
    def test_counts_non_nan_only(self):
        self.assertEqual(SF.count_at_risk([1.0, 2.0, float("nan"), 3.0, float("nan")]), 3)

    def test_all_valid(self):
        self.assertEqual(SF.count_at_risk([1.0, 2.0, 3.0]), 3)

    def test_all_nan(self):
        self.assertEqual(SF.count_at_risk([float("nan"), float("nan")]), 0)


class TestHolmCorrection(unittest.TestCase):
    def test_single_significant_p_value_below_alpha(self):
        self.assertEqual(SF.holm_correction([0.01]), [True])

    def test_all_significant_survive_step_down(self):
        # m=3, sorted thresholds 0.05/3, 0.05/2, 0.05/1 — все p проходят свой порог
        self.assertEqual(SF.holm_correction([0.001, 0.01, 0.02]), [True, True, True])

    def test_step_down_stops_at_first_failure(self):
        # rank2 (p=0.04) проваливает порог 0.05/2=0.025 -> rank3, несмотря на
        # формально проходимый порог 0.05/1=0.05, тоже не отвергается
        self.assertEqual(SF.holm_correction([0.01, 0.04, 0.06]), [True, False, False])

    def test_none_values_excluded_from_m_and_marked_false(self):
        # m должно считаться как 2 (не 3) — иначе порог для 0.02 был бы строже
        self.assertEqual(SF.holm_correction([0.01, None, 0.02]), [True, False, True])

    def test_nan_values_excluded_same_as_none(self):
        self.assertEqual(SF.holm_correction([0.01, float("nan"), 0.02]), [True, False, True])

    def test_empty_list_returns_empty(self):
        self.assertEqual(SF.holm_correction([]), [])

    def test_all_none_returns_all_false(self):
        self.assertEqual(SF.holm_correction([None, None]), [False, False])

    def test_ties_dont_crash_and_are_consistent(self):
        # все три p=0.03 не проходят свой собственный шаговый порог (0.05/3≈0.0167)
        self.assertEqual(SF.holm_correction([0.03, 0.03, 0.03]), [False, False, False])

    def test_order_independence_of_result_mapping(self):
        forward = SF.holm_correction([0.04, 0.01, 0.06])
        reversed_order = SF.holm_correction([0.06, 0.04, 0.01])
        # значение 0.01 — единственное значимое в обоих случаях, независимо от позиции
        self.assertEqual(forward, [False, True, False])
        self.assertEqual(reversed_order, [False, False, True])

    def test_single_p_value_equals_uncorrected_at_alpha(self):
        self.assertEqual(SF.holm_correction([0.05], alpha=0.05), [True])
        self.assertEqual(SF.holm_correction([0.051], alpha=0.05), [False])


class TestCalculateTgdThresholdDay(unittest.TestCase):
    def test_simple_linear_crossing(self):
        # V(0)=10, порог k=1.5 -> 15. Между днём 2 (V=12) и днём 4 (V=20):
        # пересечение 15 на t = 2 + (15-12)/(20-12)*(4-2) = 2.75
        time_data = [0, 2, 4]
        volumes = [10, 12, 20]
        day, censored = SF.calculate_tgd_threshold_day(time_data, volumes, k=1.5)
        self.assertAlmostEqual(day, 2.75, places=9)
        self.assertFalse(censored)

    def test_exact_crossing_at_measured_point(self):
        time_data = [0, 1, 2]
        volumes = [10, 15, 20]  # порог 15 достигается ровно в день 1
        day, censored = SF.calculate_tgd_threshold_day(time_data, volumes, k=1.5)
        self.assertAlmostEqual(day, 1.0, places=9)
        self.assertFalse(censored)

    def test_fully_censored_returns_last_day_and_flag(self):
        time_data = [0, 1, 2, 3]
        volumes = [10, 11, 12, 13]  # никогда не достигает 15
        day, censored = SF.calculate_tgd_threshold_day(time_data, volumes, k=1.5)
        self.assertAlmostEqual(day, 3.0, places=9)
        self.assertTrue(censored)

    def test_already_above_threshold_at_t0_pathological_v0(self):
        # v0 <= 0 -> v0 >= k*v0 тривиально верно при k>1
        time_data = [0, 1, 2]
        volumes = [0, 5, 10]
        day, censored = SF.calculate_tgd_threshold_day(time_data, volumes, k=1.5)
        self.assertAlmostEqual(day, 0.0, places=9)
        self.assertFalse(censored)

    def test_non_monotonic_reports_first_crossing_not_second(self):
        # V(0)=10, порог=15. Пересекает на дне 2 (15), затем падает (рецидив-регрессия),
        # затем снова растёт и пересекает повторно на дне 6. Должен вернуться ПЕРВЫЙ день.
        time_data = [0, 2, 4, 6]
        volumes = [10, 15, 8, 20]
        day, censored = SF.calculate_tgd_threshold_day(time_data, volumes, k=1.5)
        self.assertAlmostEqual(day, 2.0, places=9)
        self.assertFalse(censored)

    def test_single_data_point_pathological_v0_zero(self):
        # Единственная точка: V(0)=0 -> порог=0, 0>=0 верно тривиально (не цензурировано)
        day, censored = SF.calculate_tgd_threshold_day([5], [0], k=1.5)
        self.assertAlmostEqual(day, 5.0, places=9)
        self.assertFalse(censored)

    def test_single_data_point_below_threshold_censored(self):
        day, censored = SF.calculate_tgd_threshold_day([5], [10], k=1.5)
        self.assertAlmostEqual(day, 5.0, places=9)
        self.assertTrue(censored)

    def test_all_nan_volumes_raises(self):
        with self.assertRaises(ValueError):
            SF.calculate_tgd_threshold_day([0, 1, 2], [float("nan")] * 3, k=1.5)

    def test_nan_in_middle_skipped_like_dropna(self):
        # День 1 (NaN) пропускается, пересечение ищется между днём 0 и днём 2 напрямую
        time_data = [0, 1, 2]
        volumes = [10, float("nan"), 20]
        day, censored = SF.calculate_tgd_threshold_day(time_data, volumes, k=1.5)
        # порог 15 между (0,10) и (2,20): t = 0 + (15-10)/(20-10)*(2-0) = 1.0
        self.assertAlmostEqual(day, 1.0, places=9)
        self.assertFalse(censored)


class TestAggregateRtogGrades(unittest.TestCase):
    def test_known_scores_map_to_known_grades_per_animal(self):
        # Один день, разные животные: проверяем пороги map_to_rtog поэлементно через медиану
        # на вырожденной (одно животное) группе - медиана = сам перевод этого животного.
        cases = [(0, 0), (50, 1), (100, 1), (101, 1), (200, 1), (201, 2),
                 (400, 2), (401, 3), (600, 3), (601, 4)]
        for raw_score, expected_grade in cases:
            with self.subTest(raw_score=raw_score):
                result = SF.aggregate_rtog_grades([[raw_score]])
                self.assertEqual(result, [float(expected_grade)])

    def test_median_not_mean_for_even_split(self):
        # 2 животных: одно стабильно grade 1 (балл 50), другое стабильно grade 3 (балл 450).
        # Медиана по животным в каждый день -> 2.0 (среднее рангов 1 и 3).
        # Если бы сначала усредняли сырые баллы (50+450)/2=250 -> map_to_rtog(250)=2 - то же
        # значение здесь совпадает по случайности диапазонов; ключевая регрессия — что функция
        # действительно агрегирует ПОСЛЕ перевода, а не до (см. test_nan_animal_excluded_...).
        result = SF.aggregate_rtog_grades([[50, 50], [450, 450]])
        self.assertEqual(result, [2.0, 2.0])

    def test_nan_animal_excluded_from_median_at_that_day(self):
        # День 0: животное 2 = NaN -> медиана берётся только по животному 1 (grade 1)
        result = SF.aggregate_rtog_grades([[50, 50], [float("nan"), 450]])
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result[1], 2.0)  # день 1: оба валидны, grade 1 и grade 3 -> медиана 2.0

    def test_all_animals_nan_at_a_day_returns_nan(self):
        result = SF.aggregate_rtog_grades([[float("nan")], [float("nan")]])
        self.assertTrue(math.isnan(result[0]))

    def test_single_animal_group(self):
        result = SF.aggregate_rtog_grades([[50, 250, 450]])
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_empty_input_returns_empty(self):
        self.assertEqual(SF.aggregate_rtog_grades([]), [])

    def test_does_not_mutate_global_matplotlib_rcparams(self):
        """
        Регрессия: from_our_scale_to_rtog.py раньше на уровне модуля вызывал
        sns.set_theme(palette='Spectral') и plt.rcParams.update({font.family: 'Arial', ...}).
        aggregate_rtog_grades делает локальный импорт этого модуля при первом вызове — если бы
        эти вызовы оставались на уровне модуля, ПЕРВОЕ включение RTOG необратимо меняло бы
        шрифты/цвета ВСЕХ графиков приложения до конца сессии, даже без RTOG.
        """
        import matplotlib.pyplot as plt
        font_family_before = list(plt.rcParams['font.family'])
        cycle_before = list(plt.rcParams['axes.prop_cycle'])

        SF.aggregate_rtog_grades([[0, 100, 250], [0, 110, 260]])

        self.assertEqual(list(plt.rcParams['font.family']), font_family_before)
        self.assertEqual(list(plt.rcParams['axes.prop_cycle']), cycle_before)


class TestCalculateRtogIqr(unittest.TestCase):
    def test_known_spread_gives_exact_quartiles(self):
        # 4 животных, сырые баллы -> grades [0, 0, 4, 4] в один день
        # sorted=[0,0,4,4]: Q1 (25-й перцентиль) интерполируется между индексами 0 и 1 -> 0.0
        # Q3 (75-й перцентиль) интерполируется между индексами 2 и 3 -> 4.0
        skin_reactions = [[0], [0], [700], [700]]
        q1, q3 = SF.calculate_rtog_iqr(skin_reactions)
        self.assertAlmostEqual(q1[0], 0.0, places=9)
        self.assertAlmostEqual(q3[0], 4.0, places=9)

    def test_single_animal_q1_equals_q3_equals_value(self):
        q1, q3 = SF.calculate_rtog_iqr([[50, 250, 450]])
        self.assertEqual(q1, [1.0, 2.0, 3.0])
        self.assertEqual(q3, [1.0, 2.0, 3.0])

    def test_nan_animal_excluded(self):
        # День 0: одно животное NaN -> Q1=Q3 по оставшемуся единственному валидному животному
        q1, q3 = SF.calculate_rtog_iqr([[50, 50], [float("nan"), 450]])
        self.assertEqual(q1[0], 1.0)
        self.assertEqual(q3[0], 1.0)

    def test_empty_input_returns_two_empty_lists(self):
        self.assertEqual(SF.calculate_rtog_iqr([]), ([], []))


class TestCalculateSkinReactionPeak(unittest.TestCase):
    def test_peak_value_and_day_simple(self):
        peak_value, peak_day = SF.calculate_skin_reaction_peak([0, 1, 2, 3], [0, 5, 10, 3])
        self.assertEqual(peak_value, 10)
        self.assertEqual(peak_day, 2)

    def test_peak_ties_returns_first_occurrence(self):
        peak_value, peak_day = SF.calculate_skin_reaction_peak([0, 1, 2, 3], [5, 10, 10, 5])
        self.assertEqual(peak_value, 10)
        self.assertEqual(peak_day, 1)

    def test_nan_excluded(self):
        peak_value, peak_day = SF.calculate_skin_reaction_peak([0, 1, 2], [5, float("nan"), 8])
        self.assertEqual(peak_value, 8)
        self.assertEqual(peak_day, 2)

    def test_all_nan_raises(self):
        with self.assertRaises(ValueError):
            SF.calculate_skin_reaction_peak([0, 1], [float("nan"), float("nan")])


class TestCalculateSkinReactionDurationAboveThreshold(unittest.TestCase):
    def test_single_episode(self):
        # rises above 5 between day0(0) and day1(10), falls back between day2(10) and day3(0)
        time_data = [0, 1, 2, 3]
        values = [0, 10, 10, 0]
        duration, censored = SF.calculate_skin_reaction_duration_above_threshold(time_data, values, threshold=5)
        # вход: 0 + (5-0)/(10-0)*(1-0) = 0.5; выход: 2 + (5-10)/(0-10)*(3-2) = 2.5
        self.assertAlmostEqual(duration, 2.5 - 0.5, places=6)
        self.assertFalse(censored)

    def test_multiple_episodes_summed(self):
        # два отдельных эпизода выше порога 5: [0->10->0] и [0->10->0]
        time_data = [0, 1, 2, 3, 4, 5]
        values = [0, 10, 0, 0, 10, 0]
        duration, censored = SF.calculate_skin_reaction_duration_above_threshold(time_data, values, threshold=5)
        # episode 1: вход 0.5, выход 1.5 -> 1.0; episode 2: вход 3.5, выход 4.5 -> 1.0
        self.assertAlmostEqual(duration, 2.0, places=6)
        self.assertFalse(censored)

    def test_censored_when_still_above_at_end(self):
        time_data = [0, 1, 2]
        values = [0, 10, 10]
        duration, censored = SF.calculate_skin_reaction_duration_above_threshold(time_data, values, threshold=5)
        # вход в эпизод на 0.5, наблюдение заканчивается на дне 2 -> длительность 1.5, цензурировано
        self.assertAlmostEqual(duration, 1.5, places=6)
        self.assertTrue(censored)

    def test_never_above_threshold_returns_zero_not_censored(self):
        time_data = [0, 1, 2]
        values = [1, 2, 3]
        duration, censored = SF.calculate_skin_reaction_duration_above_threshold(time_data, values, threshold=5)
        self.assertEqual(duration, 0.0)
        self.assertFalse(censored)

    def test_empty_input_returns_zero(self):
        duration, censored = SF.calculate_skin_reaction_duration_above_threshold([], [], threshold=5)
        self.assertEqual(duration, 0.0)
        self.assertFalse(censored)


class TestCalculateSkinReactionTimeToNormalization(unittest.TestCase):
    def test_simple_resolution(self):
        time_data = [0, 1, 2, 3]
        values = [0, 5, 5, 0.5]  # порог по умолчанию 1.0, спад между днём 2(5) и днём 3(0.5)
        day, censored = SF.calculate_skin_reaction_time_to_normalization(time_data, values)
        # t = 2 + (1-5)/(0.5-5)*(3-2) = 2 + (-4)/(-4.5) = 2.8889
        self.assertAlmostEqual(day, 2 + (1 - 5) / (0.5 - 5) * (3 - 2), places=6)
        self.assertFalse(censored)

    def test_transient_dip_not_counted_as_normalization(self):
        # Поднимается, КРАТКОВРЕМЕННО проваливается ниже порога в середине пика, снова
        # поднимается, затем устойчиво спадает в конце. Должна вернуться ПОСЛЕДНЯЯ точка спада,
        # а не транзиторный провал на дне 3.
        time_data = [0, 1, 2, 3, 4, 5, 6]
        values = [0, 2, 3, 0.5, 3, 2, 0.5]
        day, censored = SF.calculate_skin_reaction_time_to_normalization(time_data, values)
        expected = 5 + (1 - 2) / (0.5 - 2) * (6 - 5)
        self.assertAlmostEqual(day, expected, places=6)
        self.assertFalse(censored)
        # Наивный алгоритм первого пересечения вернул бы день ~2.8 (транзиторный провал) —
        # явно проверяем, что результат НЕ в этой ранней зоне.
        self.assertGreater(day, 4.0)

    def test_never_achieved_censored(self):
        time_data = [0, 1, 2]
        values = [2, 3, 4]
        day, censored = SF.calculate_skin_reaction_time_to_normalization(time_data, values)
        self.assertAlmostEqual(day, 2.0, places=9)
        self.assertTrue(censored)

    def test_never_above_threshold_normalized_from_start(self):
        time_data = [0, 1, 2]
        values = [0.2, 0.5, 0.8]
        day, censored = SF.calculate_skin_reaction_time_to_normalization(time_data, values)
        self.assertAlmostEqual(day, 0.0, places=9)
        self.assertFalse(censored)

    def test_all_nan_raises(self):
        with self.assertRaises(ValueError):
            SF.calculate_skin_reaction_time_to_normalization([0, 1], [float("nan"), float("nan")])


if __name__ == "__main__":
    unittest.main()
