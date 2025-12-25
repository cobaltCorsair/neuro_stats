# Окно для отображения легенды отдельно от графика

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QTextEdit, QHBoxLayout, QFileDialog, QScrollArea
from PyQt6.QtCore import Qt, QStandardPaths
from PyQt6.QtGui import QFont, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import io
from . import graph_manager


class LegendWindow(QDialog):
    """Окно для отображения легенды отдельно от основного графика"""
    
    def __init__(self, legend_data=None, parent=None):
        super().__init__(parent)
        self.legend_data = legend_data or []
        self.setWindowTitle("Легенда")
        self.setModal(False)
        self.setMinimumSize(400, 300)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Настройка интерфейса окна легенды"""
        layout = QVBoxLayout()
        
        # Заголовок
        title_label = QLabel("Легенда графика")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Область для отображения легенды
        self.legend_display = QTextEdit()
        self.legend_display.setReadOnly(True)
        self.legend_display.setMinimumHeight(200)
        layout.addWidget(self.legend_display)
        
        # Кнопки управления
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Сохранить как текст")
        self.save_button.clicked.connect(self.save_legend)
        button_layout.addWidget(self.save_button)
        
        self.save_image_button = QPushButton("Сохранить как изображение")
        self.save_image_button.clicked.connect(self.save_legend_as_image)
        button_layout.addWidget(self.save_image_button)
        
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Отображаем данные легенды, если они есть
        if self.legend_data:
            self.update_legend_display()
    
    def update_legend_display(self):
        """Обновляет отображение легенды"""
        if not self.legend_data:
            self.legend_display.setText("Нет данных для отображения")
            return
            
        legend_text = "Элементы легенды:\n\n"
        for i, item in enumerate(self.legend_data, 1):
            if isinstance(item, dict):
                # Если это словарь с информацией о линии
                label = item.get('label', f'Элемент {i}')
                color = item.get('color', 'black')
                marker = item.get('marker', 'o')
                linestyle = item.get('linestyle', '-')
                legend_text += f"{i}. {label}\n"
                legend_text += f"   Цвет: {color}, Маркер: {marker}, Стиль линии: {linestyle}\n\n"
            elif isinstance(item, str):
                legend_text += f"{i}. {item}\n"
            else:
                legend_text += f"{i}. {str(item)}\n"
        
        self.legend_display.setText(legend_text)
    
    def set_legend_data(self, legend_data):
        """Устанавливает новые данные легенды"""
        self.legend_data = legend_data
        self.update_legend_display()
    
    def save_legend(self):
        """Сохраняет легенду в текстовый файл"""
        if not self.legend_data:
            return
            
        from PyQt6.QtWidgets import QFileDialog
        from PyQt6.QtCore import QStandardPaths
        
        # Предлагаем сохранить в папку Downloads
        default_path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить легенду",
            f"{default_path}/legend.txt",
            "Текстовые файлы (*.txt);;Все файлы (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("Легенда графика\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(self.legend_display.toPlainText())
                self.parent().show_message("Легенда успешно сохранена!", "Успех")
            except Exception as e:
                self.parent().show_message(f"Ошибка при сохранении: {str(e)}", "Ошибка")

    def save_legend_as_image(self):
        """Сохраняет легенду как изображение"""
        if not self.legend_data:
            return
            
        from PyQt6.QtWidgets import QFileDialog
        from PyQt6.QtCore import QStandardPaths
        from PyQt6.QtGui import QPixmap, QPainter, QFont
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        # Предлагаем сохранить в папку Downloads
        default_path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить легенду как изображение",
            f"{default_path}/legend.png",
            "PNG файлы (*.png);;JPEG файлы (*.jpg);;Все файлы (*)"
        )
        
        if file_path:
            try:
                # Создаем фигуру matplotlib только для легенды
                fig, ax = plt.subplots(figsize=(6, 8))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                # Создаем элементы легенды
                legend_elements = []
                for item in self.legend_data:
                    if isinstance(item, dict):
                        label = item.get('label', 'Элемент')
                        color = item.get('color', 'black')
                        marker = item.get('marker', 'o')
                        linestyle = item.get('linestyle', '-')
                        
                        # Создаем элемент легенды
                        if item.get('type') == 'patch':
                            patch = mpatches.Patch(color=color, label=label)
                        else:
                            patch = plt.Line2D([], [], color=color, marker=marker, 
                                             linestyle=linestyle, label=label)
                        legend_elements.append(patch)
                
                if legend_elements:
                    # Создаем легенду
                    legend = ax.legend(handles=legend_elements, loc='center', 
                                     frameon=True, fancybox=True, shadow=True)
                    legend.set_bbox_to_anchor((0.5, 0.5))
                    
                    # Убираем оси
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    # Сохраняем изображение
                    plt.tight_layout()
                    plt.savefig(file_path, dpi=300, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                    plt.close(fig)
                
                self.parent().show_message("Легенда сохранена как изображение!", "Успех")
                
            except Exception as e:
                self.parent().show_message(f"Ошибка при сохранении изображения: {str(e)}", "Ошибка")


class LegendPreviewWindow(QDialog):
    """Окно для предпросмотра легенды как изображения с возможностью сохранения"""

    def __init__(self, figure=None, parent=None):
        super().__init__(parent)
        self.figure = figure
        self.legend_pixmap = None
        self.setWindowTitle("Предпросмотр легенды")
        self.setModal(False)
        self.setMinimumSize(400, 300)

        self.setup_ui()

        if self.figure:
            self.generate_legend_image()

    def setup_ui(self):
        """Настройка интерфейса окна предпросмотра"""
        layout = QVBoxLayout()

        # Заголовок
        title_label = QLabel("Предпросмотр легенды")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Область прокрутки для изображения легенды
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(300)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(False)
        scroll_area.setWidget(self.image_label)

        layout.addWidget(scroll_area)

        # Кнопки управления
        button_layout = QHBoxLayout()

        self.save_button = QPushButton("Сохранить как изображение")
        self.save_button.clicked.connect(self.save_legend_image)
        button_layout.addWidget(self.save_button)

        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def generate_legend_image(self):
        """Генерирует изображение легенды из GraphVisualizer или напрямую из figure"""
        try:
            # Получаем последний визуализатор
            visualizer = graph_manager.get_last_visualizer()

            # Если нет visualizer, пытаемся извлечь легенду из figure напрямую
            if not visualizer:
                if self.figure:
                    self._generate_legend_from_figure()
                else:
                    self.image_label.setText("Нет данных легенды для отображения")
                return
            legend_groups = []

            # Группа 1: Основная легенда из self.lines
            main_elements = []
            if visualizer.lines:
                for line in visualizer.lines:
                    label = line.get_label()
                    if label and not label.startswith('_'):
                        main_elements.append(
                            plt.Line2D([], [],
                                     color=line.get_color(),
                                     marker=line.get_marker(),
                                     linestyle=line.get_linestyle(),
                                     linewidth=line.get_linewidth(),
                                     markersize=8,
                                     label=label)
                        )
            if main_elements:
                # Получаем заголовок основной легенды, если он сохранён
                main_title = getattr(visualizer, 'main_legend_title', '')
                legend_groups.append((main_title, main_elements))

            # Группа 2: Дополнительные легенды из self.legend_info
            if visualizer.legend_info:
                for legend_data in visualizer.legend_info:
                    labels, title, loc, display_marker = legend_data
                    info_elements = []

                    # Добавляем элементы легенды
                    for i, label in enumerate(labels):
                        if display_marker and i < len(visualizer.lines):
                            # С маркером
                            line = visualizer.lines[i]
                            info_elements.append(
                                plt.Line2D([], [],
                                         color=line.get_color(),
                                         marker=line.get_marker(),
                                         linestyle='',
                                         linewidth=0,
                                         markersize=8,
                                         label=label)
                            )
                        else:
                            # Без маркера (текстовый элемент)
                            info_elements.append(
                                plt.Line2D([], [],
                                         color='black',
                                         marker='',
                                         linestyle='',
                                         linewidth=0,
                                         markersize=0,
                                         label=label)
                            )

                    if info_elements:
                        legend_groups.append((title if title else '', info_elements))

            # Группа 3: Легенда AUC из self.aucs
            if visualizer.aucs:
                auc_elements = []
                for i, auc in enumerate(visualizer.aucs):
                    if i < len(visualizer.lines):
                        line = visualizer.lines[i]
                        auc_elements.append(
                            plt.Line2D([], [],
                                     color=line.get_color(),
                                     marker=line.get_marker(),
                                     linestyle='',
                                     linewidth=0,
                                     markersize=8,
                                     label=f"AUC: {auc:.2f}")
                        )
                if auc_elements:
                    legend_groups.append(('Площадь под кривой', auc_elements))

            if not legend_groups:
                self.image_label.setText("Нет данных легенды для отображения")
                return

            # Создаем фигуру с несколькими subplot'ами (вертикально)
            num_legends = len(legend_groups)
            fig, axes = plt.subplots(num_legends, 1, figsize=(8, max(2, num_legends * 2)))

            # Если только одна легенда, axes не будет списком
            if num_legends == 1:
                axes = [axes]

            # Создаем отдельную легенду в каждом subplot
            for idx, (title, elements) in enumerate(legend_groups):
                ax = axes[idx]
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')

                # Создаем легенду
                legend = ax.legend(handles=elements,
                                 loc='center',
                                 frameon=True,
                                 fancybox=True,
                                 shadow=False,
                                 fontsize=12,
                                 ncol=1,
                                 title=title if title else None)

                # Настраиваем заголовок легенды
                if title:
                    legend.get_title().set_fontsize(13)
                    legend.get_title().set_fontweight('bold')

            # Сохраняем изображение в буфер
            buf = io.BytesIO()
            plt.tight_layout(pad=1.0)
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)

            buf.seek(0)
            self.legend_pixmap = QPixmap()
            self.legend_pixmap.loadFromData(buf.getvalue())

            # Отображаем изображение
            self.image_label.setPixmap(self.legend_pixmap)

        except Exception as e:
            self.image_label.setText(f"Ошибка при генерации легенды: {str(e)}")

    def _generate_legend_from_figure(self):
        """Извлекает и отображает легенду напрямую из matplotlib figure"""
        try:
            if not self.figure:
                self.image_label.setText("Нет фигуры для извлечения легенды")
                return
            
            # Получаем все оси фигуры
            axes = self.figure.get_axes()
            if not axes:
                self.image_label.setText("Нет данных легенды для отображения")
                return
            
            # Извлекаем легенды из всех осей
            legend_elements = []
            for ax in axes:
                legend = ax.get_legend()
                if legend:
                    # Извлекаем handles и labels из легенды
                    handles = legend.legendHandles
                    labels = [t.get_text() for t in legend.get_texts()]
                    
                    for handle, label in zip(handles, labels):
                        if label and not label.startswith('_'):
                            legend_elements.append((handle, label))
            
            if not legend_elements:
                self.image_label.setText("Нет данных легенды для отображения")
                return
            
            # Создаем новую фигуру только для легенды
            fig, ax = plt.subplots(figsize=(8, max(2, len(legend_elements) * 0.5)))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Извлекаем только handles и labels
            handles = [h for h, l in legend_elements]
            labels = [l for h, l in legend_elements]
            
            # Создаем легенду
            legend = ax.legend(handles=handles,
                             labels=labels,
                             loc='center',
                             frameon=True,
                             fancybox=True,
                             shadow=False,
                             fontsize=12,
                             ncol=1)
            
            # Сохраняем изображение в буфер
            buf = io.BytesIO()
            plt.tight_layout(pad=1.0)
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            buf.seek(0)
            self.legend_pixmap = QPixmap()
            self.legend_pixmap.loadFromData(buf.getvalue())
            
            # Отображаем изображение
            self.image_label.setPixmap(self.legend_pixmap)
            
        except Exception as e:
            self.image_label.setText(f"Ошибка при генерации легенды из figure: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_legend_image(self):
        """Сохраняет изображение легенды в файл"""
        if not self.legend_pixmap:
            return

        # Предлагаем сохранить в папку Downloads
        default_path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить легенду как изображение",
            f"{default_path}/legend.png",
            "PNG файлы (*.png);;JPEG файлы (*.jpg);;Все файлы (*)"
        )

        if file_path:
            try:
                self.legend_pixmap.save(file_path)
                if self.parent():
                    self.parent().show_message("Легенда успешно сохранена!", "Успех")
            except Exception as e:
                if self.parent():
                    self.parent().show_message(f"Ошибка при сохранении: {str(e)}", "Ошибка")


class LegendManager:
    """Менеджер для управления легендами графиков"""

    def __init__(self):
        self.legend_window = None
        self.legend_preview_window = None
        self.current_legend_data = []
    
    def create_legend_window(self, parent=None):
        """Создает новое окно легенды"""
        if self.legend_window is None or not self.legend_window.isVisible():
            self.legend_window = LegendWindow(parent=parent)
        return self.legend_window
    
    def extract_legend_from_figure(self, figure):
        """Извлекает данные легенды из matplotlib фигуры (все легенды)"""
        legend_data = []

        if figure:
            axes = figure.get_axes()
            for ax in axes:
                # Сначала пытаемся извлечь элементы из существующих легенд
                legends = []

                # Получаем все легенды через artists (основной способ, так как они добавляются через add_artist)
                for artist in ax.artists:
                    # Проверяем тип более надежным способом
                    if type(artist).__name__ == 'Legend':
                        legends.append(artist)

                # Также проверяем основную легенду (на всякий случай)
                if ax.get_legend() is not None and ax.get_legend() not in legends:
                    legends.append(ax.get_legend())

                # Извлекаем элементы из всех найденных легенд
                if legends:
                    for legend in legends:
                        for handle, label in zip(legend.legendHandles, legend.get_texts()):
                            label_text = label.get_text()
                            if label_text and not label_text.startswith('_'):
                                # Определяем тип handle
                                handle_type = type(handle).__name__

                                # Обрабатываем Line2D (включая текстовые элементы без маркеров)
                                if handle_type == 'Line2D':
                                    color = handle.get_color() if hasattr(handle, 'get_color') else 'black'
                                    marker = handle.get_marker() if hasattr(handle, 'get_marker') else 'o'
                                    linestyle = handle.get_linestyle() if hasattr(handle, 'get_linestyle') else '-'
                                    linewidth = handle.get_linewidth() if hasattr(handle, 'get_linewidth') else 1

                                    # Если это текстовый элемент, помечаем его
                                    if (color == 'none' or color == (0, 0, 0, 0)) and marker in [None, 'None', ''] and linestyle in ['None', '']:
                                        legend_item = {
                                            'label': label_text,
                                            'color': 'white',
                                            'marker': '',
                                            'linestyle': '',
                                            'linewidth': 0,
                                            'type': 'text_only'
                                        }
                                    else:
                                        legend_item = {
                                            'label': label_text,
                                            'color': color,
                                            'marker': marker,
                                            'linestyle': linestyle,
                                            'linewidth': linewidth
                                        }
                                # Обрабатываем Patch
                                elif hasattr(handle, 'get_facecolor'):
                                    legend_item = {
                                        'label': label_text,
                                        'color': handle.get_facecolor(),
                                        'type': 'patch'
                                    }
                                # Для всех остальных типов
                                else:
                                    legend_item = {
                                        'label': label_text,
                                        'color': 'white',
                                        'marker': '',
                                        'linestyle': '',
                                        'linewidth': 0,
                                        'type': 'text_only'
                                    }
                                legend_data.append(legend_item)
                else:
                    # Если легенд нет, получаем данные из линий напрямую
                    lines = ax.get_lines()
                    for line in lines:
                        if line.get_label() and not line.get_label().startswith('_'):
                            legend_item = {
                                'label': line.get_label(),
                                'color': line.get_color(),
                                'marker': line.get_marker(),
                                'linestyle': line.get_linestyle(),
                                'linewidth': line.get_linewidth()
                            }
                            legend_data.append(legend_item)

                    # Получаем патчи (для столбчатых диаграмм и т.д.)
                    patches = ax.patches
                    seen_labels = set()
                    for patch in patches:
                        if hasattr(patch, 'get_label'):
                            label = patch.get_label()
                            if label and not label.startswith('_') and label not in seen_labels:
                                seen_labels.add(label)
                                legend_item = {
                                    'label': label,
                                    'color': patch.get_facecolor(),
                                    'type': 'patch'
                                }
                                legend_data.append(legend_item)

        self.current_legend_data = legend_data
        return legend_data
    
    def show_legend_window(self, figure=None, parent=None):
        """Показывает окно легенды с данными из фигуры"""
        legend_data = []

        if figure:
            legend_data = self.extract_legend_from_figure(figure)

        window = self.create_legend_window(parent)
        window.set_legend_data(legend_data)
        window.show()
        window.raise_()
        window.activateWindow()

        return window

    def show_legend_preview(self, figure=None, parent=None):
        """Показывает окно предпросмотра легенды как изображения"""
        if self.legend_preview_window is None or not self.legend_preview_window.isVisible():
            self.legend_preview_window = LegendPreviewWindow(figure, parent)
        else:
            # Обновляем фигуру в существующем окне
            self.legend_preview_window.figure = figure
            self.legend_preview_window.generate_legend_image()

        self.legend_preview_window.show()
        self.legend_preview_window.raise_()
        self.legend_preview_window.activateWindow()

        return self.legend_preview_window
