from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel


class ScalingImageLabel(QLabel):
    """
    QLabel, хранящий исходный pixmap в полном размере и пересчитывающий масштаб
    (с сохранением пропорций) при каждом изменении размера — иначе при сужении
    окна картинка не уменьшается, а обрезается (setFixedSize/обычный setPixmap
    не реагируют на последующий resize виджета).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._original_pixmap: Optional[QPixmap] = None
        self.setMinimumSize(400, 260)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_original_pixmap(self, pixmap: Optional[QPixmap]):
        self._original_pixmap = pixmap
        if pixmap is None or pixmap.isNull():
            self.clear()
        else:
            self._rescale()

    def sizeHint(self):
        if self._original_pixmap is not None and not self._original_pixmap.isNull():
            return self._original_pixmap.size()
        return super().sizeHint()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        scaled = self._original_pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


# Рамка вокруг ScalingImageLabel по умолчанию растягивается на всю ширину диалога
# (stretch=1 в QVBoxLayout), а пропорции исходного изображения сохраняются — если
# доступная высота не соответствует широкой рамке, внутри неё остаётся пустое поле
# вокруг маленькой картинки. Эти две функции вместо этого считают размер рамки
# заново при каждом resize диалога, чтобы она точно облегала текущий масштаб
# картинки, без пустых полей внутри границы.
PLOT_FRAME_MARGIN = 24  # 12px с каждой стороны — должно совпадать с setContentsMargins рамки


def measure_other_content_height(dialog, plot_frame) -> int:
    """
    Временно снимает ограничения размера с рамки графика, даёт диалогу принять
    естественный размер (dialog.resize(dialog.sizeHint())) и измеряет высоту,
    которую забирает ВСЁ ОСТАЛЬНОЕ содержимое диалога (не график).
    Вызывать после того, как pixmap уже установлен и диалог показан/имеет layout.
    """
    plot_frame.setMinimumSize(0, 0)
    plot_frame.setMaximumSize(16777215, 16777215)
    dialog.resize(dialog.sizeHint())
    return dialog.height() - plot_frame.height()


def fit_plot_frame_to_available_space(dialog, plot_frame, plot_label, other_content_height: Optional[int]):
    """
    Пересчитывает фиксированный размер рамки графика так, чтобы она точно облегала
    масштабированную (с сохранением пропорций) картинку при текущем размере диалога —
    без пустых полей внутри границы. Вызывать из переопределённого resizeEvent диалога.
    """
    if other_content_height is None:
        return
    original = plot_label._original_pixmap
    if original is None or original.isNull():
        return

    aspect_ratio = original.width() / max(1, original.height())
    available_width = max(100, dialog.width() - 2 * PLOT_FRAME_MARGIN)
    available_height = max(100, dialog.height() - other_content_height - PLOT_FRAME_MARGIN)

    width = available_width
    height = int(width / aspect_ratio)
    if height > available_height:
        height = available_height
        width = int(height * aspect_ratio)

    # не уходим ниже собственного минимума label — иначе Qt всё равно не даст ему
    # сжаться настолько, и картинка "вылезет" за пределы рамки, которую мы только
    # что зафиксировали меньше этого минимума
    label_min = plot_label.minimumSize()
    width = max(width, label_min.width())
    height = max(height, label_min.height())

    plot_frame.setFixedSize(width + PLOT_FRAME_MARGIN, height + PLOT_FRAME_MARGIN)
