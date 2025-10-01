# Окно для отображения легенды отдельно от графика

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QTextEdit, QHBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


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


class LegendManager:
    """Менеджер для управления легендами графиков"""
    
    def __init__(self):
        self.legend_window = None
        self.current_legend_data = []
    
    def create_legend_window(self, parent=None):
        """Создает новое окно легенды"""
        if self.legend_window is None or not self.legend_window.isVisible():
            self.legend_window = LegendWindow(parent=parent)
        return self.legend_window
    
    def extract_legend_from_figure(self, figure):
        """Извлекает данные легенды из matplotlib фигуры"""
        legend_data = []
        
        if figure:
            axes = figure.get_axes()
            for ax in axes:
                # Получаем данные из линий напрямую (самый надежный способ)
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
                for patch in patches:
                    if hasattr(patch, 'get_label') and patch.get_label() and not patch.get_label().startswith('_'):
                        legend_item = {
                            'label': patch.get_label(),
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
