from plantuml import PlantUML

# URL сервера PlantUML
server = PlantUML(url='http://www.plantuml.com/plantuml/img/')

uml_code = """
@startuml
skinparam defaultFontName Arial
skinparam shadowing false

skinparam component {
  BackgroundColor LightBlue
  ArrowColor DodgerBlue
  BorderColor DarkBlue
}

skinparam class {
  BackgroundColor Wheat
  ArrowColor Sienna
  BorderColor SandyBrown
}

skinparam usecase {
  BackgroundColor LightYellow
  BorderColor DarkOrange
}

!define RECTANGLE(x) component x <<function>>
allow_mixing

RECTANGLE(process_skin_data_excel)
RECTANGLE(process_tumor_data_excel)
RECTANGLE(save_plot)
RECTANGLE(subscriptify)
RECTANGLE(format_experiment_params)
RECTANGLE(custom_fill_between)

class TumorDataVisualizer {
    +plot_tumor_volumes_single_graph()
    +plot_relative_tumor_volumes_single_graph()
    +plot_mean_tumor_volume()
    +plot_average_relative_tumor_volume()
    +plot_mean_relative_mean_tumor_volume()
}

class SkinReactionsVisualizer {
    +plot_skin_reactions()
    +plot_mean_skin_reactions()
    +plot_multiple_experiments()
}

class TumorDataComparator {
    +compare_tumor_volumes()
    +compare_relative_tumor_volumes()
}

class TumorDataComparatorAdvanced {
    +compare_mean_volumes()
    +compare_relative_volumes()
    +compare_control_and_experiment()
    +compare_tumor_growth_inhibition_with_multiple_experiments()
    +create_tumor_growth_inhibition_table()
}

class GraphVisualizer {
    +setup_figure()
    {static} +prepare_and_add_data_to_graph()
    {static} +prepare_mann_whitney_test()
    {static} +perform_mann_whitney_test()
    {static} +add_significance_annotation()
    +add_plot()
    +finalize_figure()
    +add_legend()
    +update_axes_limits()
}

class TumorDataProcessor {
    +get_mean_tumor_volumes()
    +get_relative_tumor_volumes()
    +get_mean_relative_tumor_volumes()
}

class SkinReactionsDataProcessor {
    +get_mean_skin_reactions()
}

class MatplotlibConfigurator {
    +apply_custom_styles()
    +restore_original_styles()
}

class ExtractOutliers {
    +remove_outliers()
    +remove_outliers_iqr()
    +remove_outliers_grubbs()
    +remove_outliers_elliptic_envelope()
    +remove_outliers_isolation_forest()
    +remove_outliers_mahalanobis()
    +exclude_rats()
}

class SupportingFunctions {
    {static} +calculate_std_dev()
    {static} +calculate_error_margin()
    {static} +interpolate_data_to_common_timepoints()
    {static} +calculate_auc()
    {static} +calculate_tumor_growth_inhibition()
    {static} +trim_data_to_timepoint()
    {static} +normalize_time_data()
    {static} +trim_data_to_common_length()
    {static} +normalize_time_data_min()
}

class ControlGroupVisualizer {
    +process_excel()
}

TumorDataVisualizer --> GraphVisualizer : uses >
SkinReactionsVisualizer --> GraphVisualizer : uses >
TumorDataComparator --> GraphVisualizer : uses >
TumorDataComparatorAdvanced --> GraphVisualizer : uses >
TumorDataVisualizer --> TumorDataProcessor : uses >
SkinReactionsVisualizer --> SkinReactionsDataProcessor : uses >
TumorDataVisualizer ..|> ControlGroupVisualizer : <<extends>>
ExtractOutliers --> TumorDataVisualizer : modifies >
ExtractOutliers --> SkinReactionsVisualizer : modifies >
TumorDataProcessor ..> process_tumor_data_excel : uses >
SkinReactionsDataProcessor ..> process_skin_data_excel : uses >
TumorDataVisualizer ..> SupportingFunctions : uses >
SkinReactionsVisualizer ..> SupportingFunctions : uses >
TumorDataComparator ..> SupportingFunctions : uses >
TumorDataComparatorAdvanced ..> SupportingFunctions : uses >
GraphVisualizer ..> save_plot : uses >
GraphVisualizer ..> format_experiment_params : uses >
TumorDataComparatorAdvanced ..> custom_fill_between : uses >
TumorDataVisualizer ..> custom_fill_between : uses >
GraphVisualizer ..> subscriptify : uses >
@enduml
"""

# Генерация и сохранение UML-диаграммы
with open("diagram.uml", "w") as file:
    file.write(uml_code)

# Отправка UML-кода на сервер PlantUML для генерации изображения
server.processes_file("diagram.uml")
