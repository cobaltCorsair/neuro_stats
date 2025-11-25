# Хранение меток крыс и имён файлов
rat_labels_with_indices = []  # Список кортежей (метка крысы, имя файла)


def register_rat_labels(labels, file_name):
    """
    Регистрирует метки крыс с указанием имени файла.

    Args:
        labels (List[str]): Список меток крыс.
        file_name (str): Имя файла, из которого были извлечены метки.
    """
    global rat_labels_with_indices
    rat_labels_with_indices.extend([(label, file_name) for label in labels])


def get_rat_labels():
    """
    Возвращает текущие метки крыс в виде списка кортежей (метка крысы, имя файла).
    """
    return rat_labels_with_indices


def clear_rat_labels():
    """
    Очищает метки крыс.
    """
    global rat_labels_with_indices
    rat_labels_with_indices = []
