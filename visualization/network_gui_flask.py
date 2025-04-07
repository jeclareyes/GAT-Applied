# visualization/network_gui_flask.py
from flask import Flask, render_template_string, request
from visualization.network_viz import NetworkVisualizer_Pyvis
import torch

app = Flask(__name__)

# Datos dummy para ejemplo – en producción se usarían datos reales
nodes = torch.tensor([[597., 446.],
                       [631., 516.],
                       [528., 418.],
                       [540., 619.],
                       [426., 444.]])
links = torch.tensor([[0, 1, 2, 3],
                      [1, 2, 3, 4]])
estimated_flows = torch.tensor([0., 100., 150., 200.])
observed_flows = torch.tensor([50., 120., 140., 210.])
viz = NetworkVisualizer_Pyvis(nodes, links, estimated_flows, observed_flows)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        viz.update_config(
            show_node_labels = request.form.get("show_node_labels") == "on",
            node_label_font_size = int(request.form.get("node_label_font_size")),
            node_size_multiplier = float(request.form.get("node_size_multiplier")),
            node_default_color = request.form.get("node_default_color"),
            edge_label_mode = request.form.get("edge_label_mode"),
            show_edge_labels = request.form.get("show_edge_labels") == "on",
            base_edge_width = float(request.form.get("base_edge_width")),
            edge_width_scaling = float(request.form.get("edge_width_scaling")),
            flow_tolerance = float(request.form.get("flow_tolerance")),
            gravitational_constant = int(request.form.get("gravitational_constant")),
            spring_length = int(request.form.get("spring_length")),
            bidirectional = request.form.get("bidirectional") == "on",
            export_filepath = request.form.get("export_filepath")
        )
    html_data = viz.draw()
    template = """
    <html>
    <head>
      <title>Visualizador de Red</title>
    </head>
    <body>
      <h1>Configuración de Visualización</h1>
      <form method="POST">
        <label>Mostrar etiquetas nodos:
          <input type="checkbox" name="show_node_labels" {% if viz.config.show_node_labels %}checked{% endif %}>
        </label><br>
        <label>Tamaño fuente nodos:
          <input type="number" name="node_label_font_size" value="{{ viz.config.node_label_font_size }}">
        </label><br>
        <label>Multiplicador tamaño nodos:
          <input type="number" step="0.1" name="node_size_multiplier" value="{{ viz.config.node_size_multiplier }}">
        </label><br>
        <label>Color nodos:
          <input type="color" name="node_default_color" value="{{ viz.config.node_default_color }}">
        </label><br>
        <label>Modo etiquetas aristas:
          <select name="edge_label_mode">
            <option value="combined" {% if viz.config.edge_label_mode=='combined' %}selected{% endif %}>Combined</option>
            <option value="estimated" {% if viz.config.edge_label_mode=='estimated' %}selected{% endif %}>Estimated</option>
            <option value="observed" {% if viz.config.edge_label_mode=='observed' %}selected{% endif %}>Observed</option>
          </select>
        </label><br>
        <label>Mostrar etiquetas aristas:
          <input type="checkbox" name="show_edge_labels" {% if viz.config.show_edge_labels %}checked{% endif %}>
        </label><br>
        <label>Ancho base aristas:
          <input type="number" step="0.1" name="base_edge_width" value="{{ viz.config.base_edge_width }}">
        </label><br>
        <label>Escala ancho aristas:
          <input type="number" step="0.001" name="edge_width_scaling" value="{{ viz.config.edge_width_scaling }}">
        </label><br>
        <label>Tolerancia flujo:
          <input type="number" step="0.5" name="flow_tolerance" value="{{ viz.config.flow_tolerance }}">
        </label><br>
        <label>Gravedad:
          <input type="number" name="gravitational_constant" value="{{ viz.config.gravitational_constant }}">
        </label><br>
        <label>Longitud resorte:
          <input type="number" name="spring_length" value="{{ viz.config.spring_length }}">
        </label><br>
        <label>Modo bidireccional:
          <input type="checkbox" name="bidirectional" {% if viz.config.bidirectional %}checked{% endif %}>
        </label><br>
        <label>Ruta HTML exportado:
          <input type="text" name="export_filepath" value="{{ viz.config.export_filepath }}">
        </label><br>
        <input type="submit" value="Actualizar">
      </form>
      <hr>
      <h2>Visualización de la Red</h2>
      {{ html_data|safe }}
    </body>
    </html>
    """
    from flask import render_template_string
    return render_template_string(template, viz=viz, html_data=html_data)

def main():
    app.run(debug=True, port=8501)

if __name__ == "__main__":
    main()
