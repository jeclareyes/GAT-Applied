import json
import datetime

def save_training_report(model, model_run, training_info, loss_function_type, do_train, node_metrics, edge_metrics, report_route="training_report.json"):
    """
    Guarda un reporte JSON con información del entrenamiento, estructura del modelo, flujos estimados y errores.

    Parámetros:
      - model: el modelo entrenado (se guardará su estructura vía str(model)).
      - training_info: diccionario con información de entrenamiento (número de epochs, últimas pérdidas, etc.).
      - node_metrics: lista de diccionarios con métricas por nodo (resultado de compute_node_errors).
      - edge_metrics: lista de diccionarios con métricas por enlace (resultado de compute_edge_errors).
      - report_route: ruta del archivo donde se guardará el reporte.
    """
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
             "model_run": model_run,
             "loss_function_type": loss_function_type if do_train else "N/A (No Training)",
             "training_info": training_info if do_train else {"status": "Skipped"},
             "model_structure": str(model),
             "node_metrics": node_metrics,
             "edge_metrics": edge_metrics
    }
    with open(report_route, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Reporte de entrenamiento guardado en {report_route}")
