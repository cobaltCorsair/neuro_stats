# Модуль для управления визуализаторами и положением легенды

from typing import List

visualizers = []
current_legend_position = 'best'  # По умолчанию
rat_labels = []
holm_correction_enabled = True  # Поправка Холма на множественность поточечных сравнений
show_date_in_legend = True  # Показывать "Дата: ..." в подписях экспериментов


def set_holm_correction_enabled(enabled: bool):
    global holm_correction_enabled
    holm_correction_enabled = enabled


def is_holm_correction_enabled() -> bool:
    return holm_correction_enabled


def set_show_date_in_legend(enabled: bool):
    global show_date_in_legend
    show_date_in_legend = enabled


def is_show_date_in_legend() -> bool:
    return show_date_in_legend


def build_significance_test_legend_label(test_name: str) -> List[str]:
    """
    Строки для отдельной легенды на графике, поясняющей, какой критерий значимости
    использован и что означает каждый из маркеров — без этого на графике видны только
    символы '*'/'(*)' без объяснения.

    При включённой поправке Холма на графике одновременно встречаются ДВА разных
    маркера ('*' — значимо после поправки, '(*)' — значимо только по сырому p<0.05,
    поправку не прошло), поэтому легенда обязана объяснять оба, а не только один.

    Последняя строка — постоянное предупреждение о том, что высокая внутригрупповая
    изменчивость (смешанный ответ на лечение: часть животных регрессировала, часть — нет)
    снижает мощность и Манна-Уитни, и Стьюдента одинаково, так что визуально большая разница
    средних на графике не обязана сопровождаться маркером значимости.
    """
    if holm_correction_enabled:
        lines = [
            f"* {test_name}, значимо с поправкой Холма",
            "(*) значимо только без поправки Холма",
        ]
    else:
        lines = [f"* {test_name}, без поправки Холма"]
    lines.append("Высокая внутригрупповая изменчивость (смешанный ответ) снижает мощность теста")
    return lines


def register_visualizer(visualizer):
    visualizers.append(visualizer)
    # Сразу обновляем визуализатор до актуального положения легенды
    visualizer.update_legend_position(current_legend_position)


def update_legend_position(position):
    global current_legend_position
    current_legend_position = position  # Обновляем текущее положение
    for visualizer in visualizers:
        visualizer.update_legend_position(position)


def get_current_legend_position():
    return current_legend_position


def register_rat_labels(labels):
    """
    Регистрирует метки крыс.
    """
    global rat_labels
    rat_labels = labels


def get_rat_labels():
    """
    Возвращает текущие метки крыс.
    """
    return rat_labels


def clear_rat_labels():
    """
    Очищает метки крыс.
    """
    global rat_labels
    rat_labels = []


def get_last_visualizer():
    """
    Возвращает последний зарегистрированный визуализатор.
    """
    if visualizers:
        return visualizers[-1]
    return None
