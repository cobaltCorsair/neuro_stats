class GraphManager:
    _instance = None

    def __init__(self):
        self.visualizers = []
        self.current_legend_position = 'best'  # По умолчанию

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = GraphManager()
        return cls._instance

    def register_visualizer(self, visualizer):
        self.visualizers.append(visualizer)
        print(f"Visualizer registered: {visualizer}")
        # Сразу обновляем визуализатор до актуального положения легенды
        visualizer.update_legend_position(self.current_legend_position)  # Передаем актуальное положение

    def update_legend_position(self, position):
        print(f"GraphManager updating legend position to: {position}")
        self.current_legend_position = position  # Обновляем текущее положение
        for visualizer in self.visualizers:
            print(f"Updating visualizer {visualizer} with new position: {position}")
            visualizer.update_legend_position(position)

    def get_current_legend_position(self):
        # Метод для получения текущего положения легенды
        return self.current_legend_position
