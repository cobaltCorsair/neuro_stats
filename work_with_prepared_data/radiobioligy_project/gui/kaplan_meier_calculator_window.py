import io

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QPalette, QPixmap
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QFrame, QHBoxLayout, QHeaderView, QLabel, QListView, QMessageBox,
    QPlainTextEdit, QPushButton, QStyle, QStyledItemDelegate, QStyleOptionViewItem,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from work_with_prepared_data.radiobioligy_project.data_processing.excel_data_processor import RatSurvivalEvent
from work_with_prepared_data.radiobioligy_project.gui.scaling_image_label import (
    ScalingImageLabel,
    fit_plot_frame_to_available_space,
    measure_other_content_height,
)
from work_with_prepared_data.radiobioligy_project.stats_methods.kaplan_meier import (
    format_calculation_steps,
    kaplan_meier_estimate,
    median_survival_ci,
    n_at_risk_table,
    plot_kaplan_meier,
    restricted_mean_survival_time,
    risk_table_time_points,
)

EVENT_DEATH = "Смерть"
EVENT_CENSORED = "Цензурирован"
EVENT_ALIVE = "Жив"


class _ComboPopupItemDelegate(QStyledItemDelegate):
    """Исключает нативную чёрную подсветку в popup-списках QComboBox.
    Это отдельное top-level окно (QDialog), стиль MainWindow сюда не каскадируется."""

    _base_color = QColor('#F0F5FA')
    _text_color = QColor('#243040')
    _highlight_color = QColor('#C5D9EE')
    _highlight_text_color = QColor('#1A3050')

    def paint(self, painter, option, index):
        item_option = QStyleOptionViewItem(option)
        self.initStyleOption(item_option, index)

        is_highlighted = bool(
            item_option.state & QStyle.StateFlag.State_MouseOver
            or item_option.state & QStyle.StateFlag.State_Selected
        )
        background = self._highlight_color if is_highlighted else self._base_color
        foreground = self._highlight_text_color if is_highlighted else self._text_color

        painter.fillRect(item_option.rect, background)
        item_option.backgroundBrush = QBrush(background)
        item_option.palette.setColor(QPalette.ColorRole.Base, background)
        item_option.palette.setColor(QPalette.ColorRole.Window, background)
        item_option.palette.setColor(QPalette.ColorRole.Text, foreground)
        item_option.palette.setColor(QPalette.ColorRole.WindowText, foreground)
        item_option.state &= ~QStyle.StateFlag.State_Selected
        item_option.state &= ~QStyle.StateFlag.State_MouseOver
        item_option.state &= ~QStyle.StateFlag.State_HasFocus
        super().paint(painter, item_option, index)

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(max(size.height() + 8, 30))
        return size


class KaplanMeierCalculatorWindow(QDialog):
    """
    Калькулятор Каплана-Майера с ручным вводом данных (время + событие на пациента) —
    для обучения, проверки расчёта вручную или быстрого анализа без Excel-файла.
    В отличие от KaplanMeierWindow (сравнение групп из выбранных файлов), здесь всегда
    один набор данных и явный пошаговый расчёт S(t).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("Калькулятор Каплана-Майера")
        self.setMinimumSize(550, 500)

        self.input_table = None
        self.plot_label = None
        self.plot_frame = None
        self.risk_table = None
        self.steps_text = None
        self.summary_label = None
        self._other_content_height = None
        self._setup_ui()
        self._add_input_row()
        self._add_input_row()
        self._add_input_row()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        fit_plot_frame_to_available_space(self, self.plot_frame, self.plot_label, self._other_content_height)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        title_label = QLabel("Калькулятор Каплана-Майера")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title_label)

        hint_label = QLabel(
            "Введите время наблюдения и событие для каждого пациента/животного, затем нажмите «Рассчитать»."
        )
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)

        self.input_table = QTableWidget()
        self.input_table.setColumnCount(3)
        self.input_table.setHorizontalHeaderLabels(["Метка", "Время", "Событие"])
        self.input_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.input_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.input_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.input_table.setMaximumHeight(220)
        layout.addWidget(self.input_table)

        load_button = QPushButton("Загрузить из выбранных файлов...")
        load_button.clicked.connect(self._handle_load_from_main_window)
        layout.addWidget(load_button)

        buttons_layout = QHBoxLayout()
        add_row_button = QPushButton("+ Добавить пациента")
        add_row_button.clicked.connect(self._add_input_row)
        remove_row_button = QPushButton("− Удалить выбранную строку")
        remove_row_button.clicked.connect(self._remove_selected_row)
        calculate_button = QPushButton("Рассчитать")
        calculate_button.setStyleSheet("font-weight: 600;")
        calculate_button.clicked.connect(self._handle_calculate)
        buttons_layout.addWidget(add_row_button)
        buttons_layout.addWidget(remove_row_button)
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(calculate_button)
        layout.addLayout(buttons_layout)

        self.plot_frame = QFrame()
        self.plot_frame.setObjectName("kmCalcPlotFrame")
        self.plot_frame.setStyleSheet(
            "#kmCalcPlotFrame { border: 1px solid #c0c0c0; border-radius: 4px; background-color: white; }"
        )
        plot_frame_layout = QVBoxLayout(self.plot_frame)
        plot_frame_layout.setContentsMargins(12, 12, 12, 12)
        self.plot_label = ScalingImageLabel()
        plot_frame_layout.addWidget(self.plot_label)
        layout.addWidget(self.plot_frame, stretch=1, alignment=Qt.AlignmentFlag.AlignHCenter)

        risk_table_label = QLabel("Число в риске")
        risk_table_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(risk_table_label)
        self.risk_table = QTableWidget()
        self._configure_readonly_table(self.risk_table)
        layout.addWidget(self.risk_table)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        steps_label = QLabel("Пошаговый расчёт S(t)")
        steps_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(steps_label)
        self.steps_text = QPlainTextEdit()
        self.steps_text.setReadOnly(True)
        self.steps_text.setStyleSheet("font-family: Consolas, monospace;")
        self.steps_text.setMaximumHeight(160)
        layout.addWidget(self.steps_text)

    @staticmethod
    def _fix_event_combo(combo_box: QComboBox) -> None:
        """Устанавливает явную палитру и делегат для popup QComboBox в QDialog.
        QDialog не наследует QSS из MainWindow, поэтому palette(base) не разрешается
        в светлый цвет и popup рендерится с чёрным фоном."""
        combo_box.setView(QListView(combo_box))
        view = combo_box.view()
        view.setMouseTracking(True)
        view.setSpacing(0)
        view.setUniformItemSizes(True)
        view.setAutoFillBackground(True)
        view.viewport().setAutoFillBackground(True)
        view.viewport().setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        view.setItemDelegate(_ComboPopupItemDelegate(view))

        palette = view.palette()
        combo_colors = {
            QPalette.ColorRole.Base: QColor('#F0F5FA'),
            QPalette.ColorRole.AlternateBase: QColor('#F0F5FA'),
            QPalette.ColorRole.Window: QColor('#F0F5FA'),
            QPalette.ColorRole.Text: QColor('#243040'),
            QPalette.ColorRole.WindowText: QColor('#243040'),
            QPalette.ColorRole.ButtonText: QColor('#243040'),
            QPalette.ColorRole.Highlight: QColor('#C5D9EE'),
            QPalette.ColorRole.HighlightedText: QColor('#1A3050'),
        }
        for color_group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
            for color_role, color in combo_colors.items():
                palette.setColor(color_group, color_role, color)
        view.setPalette(palette)
        view.viewport().setPalette(palette)
        view.setStyleSheet("""
            QAbstractItemView {
                background-color: #F0F5FA;
                color: #243040;
                border: 1px solid #AABBCC;
                outline: 0;
                selection-background-color: #C5D9EE;
                selection-color: #1A3050;
            }
            QAbstractItemView::item {
                background-color: #F0F5FA;
                color: #243040;
                padding: 4px 8px;
                min-height: 30px;
            }
            QAbstractItemView::item:hover,
            QAbstractItemView::item:selected,
            QAbstractItemView::item:selected:active,
            QAbstractItemView::item:selected:!active {
                background-color: #C5D9EE;
                color: #1A3050;
            }
        """)

    def _add_input_row(self):
        row = self.input_table.rowCount()
        self.input_table.setRowCount(row + 1)
        self.input_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        self.input_table.setItem(row, 1, QTableWidgetItem(""))
        event_combo = QComboBox()
        event_combo.addItems([EVENT_DEATH, EVENT_CENSORED, EVENT_ALIVE])
        self._fix_event_combo(event_combo)
        self.input_table.setCellWidget(row, 2, event_combo)

    def _remove_selected_row(self):
        row = self.input_table.currentRow()
        if row >= 0:
            self.input_table.removeRow(row)

    def _handle_load_from_main_window(self):
        """
        Подтягивает события из текущего выбора файлов/групп A-B в главном окне —
        той же логикой, что строит группы для KaplanMeierWindow (сравнение групп).
        Калькулятор работает с ОДНИМ набором данных, поэтому при нескольких группах
        просит выбрать, какую из них загрузить.
        """
        main_window = self.parent()
        if main_window is None or not hasattr(main_window, "build_kaplan_meier_groups_from_selection"):
            QMessageBox.information(
                self, "Калькулятор Каплана-Майера",
                "Калькулятор не привязан к главному окну — загрузка недоступна.",
            )
            return

        try:
            groups = main_window.build_kaplan_meier_groups_from_selection()
        except Exception as error:
            QMessageBox.critical(self, "Калькулятор Каплана-Майера", f"Ошибка при чтении файлов: {error}")
            return

        if not groups:
            QMessageBox.information(
                self, "Калькулятор Каплана-Майера",
                "В главном окне не выбрано ни одного файла. Отметьте файл(ы) галочкой в таблице.",
            )
            return

        if len(groups) == 1:
            group_name, events = next(iter(groups.items()))
        else:
            from PyQt6.QtWidgets import QInputDialog
            group_name, ok = QInputDialog.getItem(
                self, "Выбор группы", "Калькулятор работает с одним набором данных — выберите группу:",
                list(groups.keys()), editable=False,
            )
            if not ok:
                return
            events = groups[group_name]

        if not events:
            QMessageBox.information(self, "Калькулятор Каплана-Майера", f"В группе «{group_name}» нет данных.")
            return

        self._populate_input_table_from_events(events)

    def _populate_input_table_from_events(self, events):
        self.input_table.setRowCount(0)
        for event in events:
            row = self.input_table.rowCount()
            self.input_table.setRowCount(row + 1)
            self.input_table.setItem(row, 0, QTableWidgetItem(event.label))
            day_text = "" if event.day is None else f"{event.day:g}"
            self.input_table.setItem(row, 1, QTableWidgetItem(day_text))
            event_combo = QComboBox()
            event_combo.addItems([EVENT_DEATH, EVENT_CENSORED, EVENT_ALIVE])
            if event.event_observed:
                event_combo.setCurrentText(EVENT_DEATH)
            elif event.reason == EVENT_ALIVE:
                event_combo.setCurrentText(EVENT_ALIVE)
            else:
                event_combo.setCurrentText(EVENT_CENSORED)
            self._fix_event_combo(event_combo)
            self.input_table.setCellWidget(row, 2, event_combo)

    def _read_input_events(self):
        """
        Читает таблицу ввода в список RatSurvivalEvent.
        Бросает ValueError с понятным сообщением при первой некорректной строке.
        """
        events = []
        for row in range(self.input_table.rowCount()):
            label_item = self.input_table.item(row, 0)
            time_item = self.input_table.item(row, 1)
            event_widget = self.input_table.cellWidget(row, 2)

            time_text = time_item.text().strip() if time_item else ""
            if not time_text:
                continue  # пустая строка - пропускаем, не считаем ошибкой

            try:
                day = float(time_text.replace(",", "."))
            except ValueError:
                raise ValueError(f"Строка {row + 1}: «{time_text}» не похоже на число (время).")
            if day < 0:
                raise ValueError(f"Строка {row + 1}: время не может быть отрицательным.")

            label = label_item.text().strip() if label_item and label_item.text().strip() else str(row + 1)
            current_text = event_widget.currentText() if event_widget else EVENT_DEATH
            is_death = current_text == EVENT_DEATH
            events.append(RatSurvivalEvent(label, day, is_death, current_text))
        return events

    def _handle_calculate(self):
        try:
            events = self._read_input_events()
        except ValueError as error:
            QMessageBox.warning(self, "Калькулятор Каплана-Майера", str(error))
            return

        if not events:
            QMessageBox.information(self, "Калькулятор Каплана-Майера", "Заполните хотя бы одну строку с временем.")
            return

        km = kaplan_meier_estimate(events)

        fig, _ax = plot_kaplan_meier({"Данные": events}, show_ci=True)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        self.plot_label.set_original_pixmap(pixmap)

        self._populate_risk_table(events)
        self._update_summary(km, events)
        self.steps_text.setPlainText("\n".join(format_calculation_steps(km)) or "Нет точек для расчёта.")

        self._other_content_height = measure_other_content_height(self, self.plot_frame)
        fit_plot_frame_to_available_space(self, self.plot_frame, self.plot_label, self._other_content_height)

    def _populate_risk_table(self, events):
        time_points = risk_table_time_points({"Данные": events})
        counts = n_at_risk_table({"Данные": events}, time_points)["Данные"]

        self.risk_table.clear()
        self.risk_table.setColumnCount(len(time_points))
        self.risk_table.setHorizontalHeaderLabels([f"{t:.0f}" for t in time_points])
        self.risk_table.setRowCount(1)
        for col, n in enumerate(counts):
            item = QTableWidgetItem(str(n))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.risk_table.setItem(0, col, item)
        self.risk_table.resizeRowsToContents()
        # без этого столбцы остаются фиксированной ширины, и при растяжении окна
        # справа от последнего столбца остаётся пустая серая область
        self.risk_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._fit_table_height(self.risk_table)

    def _update_summary(self, km, events):
        n_total = sum(1 for e in events if e.day is not None)
        n_events = sum(1 for e in events if e.event_observed)
        n_censored = n_total - n_events
        median = "—" if km.median_survival is None else f"{km.median_survival:g}"
        ci_low, ci_high = median_survival_ci(km)
        if ci_low is None and ci_high is None:
            median_ci = "—"
        else:
            low_text = f"{ci_low:g}" if ci_low is not None else "?"
            high_text = f"{ci_high:g}" if ci_high is not None else "не достигнута"
            median_ci = f"[{low_text}, {high_text}]"
        rmst = restricted_mean_survival_time(km)
        final_s = km.survival[-1] if km.survival else 1.0
        final_t = km.times[-1] if km.times else 0.0

        self.summary_label.setText(
            f"N = {n_total}, событий = {n_events}, цензурировано = {n_censored}. "
            f"Медиана выживаемости: {median} (95% ДИ {median_ci}). RMST = {rmst:.3f}. "
            f"S({final_t:g}) = {final_s:.4f}."
        )

    @staticmethod
    def _configure_readonly_table(table_widget):
        table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table_widget.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table_widget.setAlternatingRowColors(True)
        table_widget.verticalHeader().setVisible(False)

    @staticmethod
    def _fit_table_height(table_widget):
        header_height = table_widget.horizontalHeader().height()
        rows_height = sum(table_widget.rowHeight(row) for row in range(table_widget.rowCount()))
        frame = 2 * table_widget.frameWidth()
        total = header_height + rows_height + frame + 4
        table_widget.setFixedHeight(max(total, header_height + frame + 4))
