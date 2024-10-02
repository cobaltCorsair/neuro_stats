# Модуль для управления визуализаторами и положением легенды

visualizers = []
current_legend_position = 'best'  # По умолчанию
rat_labels = []


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
