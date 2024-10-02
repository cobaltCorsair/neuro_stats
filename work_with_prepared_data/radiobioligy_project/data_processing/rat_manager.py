# Хранение меток крыс и индексов наборов данных
rat_labels_with_indices = []  # Список кортежей (метка крысы, индекс набора данных)

current_data_index = 0  # Индекс текущего набора данных


def register_rat_labels(labels):
    """
    Регистрирует метки крыс с указанием индекса набора данных.

    Args:
        labels (List[str]): Список меток крыс.
    """
    global rat_labels_with_indices, current_data_index
    rat_labels_with_indices.extend([(label, current_data_index) for label in labels])
    current_data_index += 1  # Увеличиваем индекс набора данных


def get_rat_labels():
    """
    Возвращает текущие метки крыс в виде списка кортежей (метка крысы, индекс набора данных).
    """
    return rat_labels_with_indices


def clear_rat_labels():
    """
    Очищает метки крыс.
    """
    global rat_labels_with_indices, current_data_index
    rat_labels_with_indices = []
    current_data_index = 0  # Сбрасываем индекс данных
