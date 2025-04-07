import os
from pyvis.network import Network

def previsualization_pyvis(data, output_file="graph_preview.html"):
    x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y

    # Crear una red dirigida con ajustes visuales
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
        directed=True
    )

    # Estilo de fuerza para nodos (aunque se desactiva con physics=False)
    net.barnes_hut(gravity=-200, central_gravity=0.3)

    # Agregar nodos con mejor estilo y tamaño de texto
    for i in range(x.size(0)):
        label = f"No.: {i}\nGen./Atr.:\n ({x[i,0].item():.2f}, {x[i,1].item():.2f})"
        x_pos = x[i, 0].item() * 100
        y_pos = x[i, 1].item() * 100
        net.add_node(
            i,
            label=label,
            x=x_pos,
            y=y_pos,
            physics=False,
            size=25,
            font={'size': 30},
            color='#97C2FC'
        )

    # Agregar aristas con etiquetas legibles y flechas
    for idx in range(edge_index.size(1)):
        source = edge_index[0, idx].item()
        target = edge_index[1, idx].item()
        attr_val = edge_attr[idx].item()
        y_val = y[idx].item()
        edge_label = f"({source},{target})\nsensor: {attr_val:.2f}\nreal: {y_val:.2f}"
        net.add_edge(
            source,
            target,
            label=edge_label,
            arrows="to",
            smooth={'enabled': True, 'type': 'curvedCW', 'roundness': 0.2},
            width=4,
            font={'size': 30},
            color='#848484'
        )

    # (Opcional) Más ajustes de visualización usando opciones en JSON
    net.set_options('''
    var options = {
      "nodes": {
        "font": {
          "size": 30
        }
      },
      "edges": {
        "font": {
          "size": 16,
          "align": "top"
        },
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 3
          }
        },
        "smooth": {
          "enabled": true
        }
      },
      "physics": {
        "enabled": false
      }
    }
    ''')

    # Guardar el archivo HTML
    html_str = net.generate_html()
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_str)

    print(f"Grafo exportado como: {output_file}")


if __name__ == '__main__':
    from data.loader import load_traffic_data
    PICKLE_FILE = "traffic_data_2.pkl"
    data = load_traffic_data(os.path.join("", PICKLE_FILE))
    previsualization_pyvis(data)
