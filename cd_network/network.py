import json
from collections import defaultdict, deque
from typing import Any, Dict, Union

import numpy as np

from .cells import cd, ee, ei, simple_ee


class Neuron:
    def __init__(self, cell_type: str, cell_id: str, params: Dict[str, Any], fs: float):
        self.cell_type = cell_type
        self.cell_id = cell_id
        self.params = params
        self.fs = fs

    def __call__(self, inputs: Dict[str, np.ndarray], *args, **kwargs) -> np.ndarray:
        excitatory_inputs = inputs.get("excitatory")
        inhibitory_inputs = inputs.get("inhibitory")

        if self.cell_type == "ei":
            return ei(
                excitatory_inputs, inhibitory_inputs, self.params["delta_s"], self.fs
            )
        elif self.cell_type == "simple_ee":
            return simple_ee(excitatory_inputs, self.params["delta_s"], self.fs)
        elif self.cell_type == "ee":
            return ee(
                excitatory_inputs,
                self.params["n_spikes"],
                self.params["delta_s"],
                self.fs,
            )
        elif self.cell_type == "cd":
            return cd(
                excitatory_inputs,
                inhibitory_inputs,
                self.params["n_spikes"],
                self.params["delta_s"],
                self.fs,
            )
        else:
            raise ValueError(f"Unknown cell type: {self.cell_type}")


class CDNetwork:
    def __init__(self, config: Union[Dict[str, Any], str]):
        """Initialize the network with a configuration dictionary or a path to a configuration JSON file."""
        self.cells = {}
        self.connections = []
        self.load_config(config)

    def load_config(self, config: Union[Dict[str, Any], str]):
        """Load and parse the configuration from a dictionary or a JSON file."""
        if isinstance(config, str):
            try:
                with open(config, "r") as file:
                    config = json.load(file)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"The configuration file {config} was not found."
                )
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format in the configuration file.")

        if not isinstance(config, dict):
            raise ValueError(
                "Configuration must be a dictionary or a path to a JSON file."
            )

        fs = config["fs"]
        for cell_config in config["cells"]:
            cell = Neuron(
                cell_type=cell_config["type"],
                cell_id=cell_config["id"],
                params=cell_config["params"],
                fs=fs,
            )
            self.cells[cell_config["id"]] = cell
        self.connections = config["connections"]

    def plot_network_connections(self):
        """
        Plot the network connections using NetworkX and Matplotlib.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.DiGraph()
        for cell_id in self.cells:
            cell_type = self.cells[cell_id].cell_type
            color_map = {"excitatory": "green", "inhibitory": "red"}
            node_color = color_map.get(cell_type, "skyblue")
            G.add_node(cell_id, label=f"({cell_type})", color=node_color)
        for conn in self.connections:
            G.add_edge(conn["source"], conn["target"], label=conn["input_type"])

        pos = nx.spring_layout(G)  # Consider changing the layout for large networks
        colors = [G.nodes[node].get("color", "white") for node in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos, node_size=5000, node_color=colors, alpha=1
        )  # edgecolors='black'

        node_labels = {
            node: f"{node} \n {G.nodes[node].get('label', '')}" for node in G.nodes()
        }
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12)

        nx.draw_networkx_edges(
            G, pos, arrowstyle="-|>", arrowsize=20, edge_color="gray"
        )
        edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

        plt.title("CDNetwork Connections")
        plt.axis("off")
        plt.show()

    def __call__(self, external_inputs: Dict[str, np.ndarray], *args, **kwargs):
        cell_outputs = {}
        cell_inputs = {
            cell_id: {"excitatory": [], "inhibitory": []}
            for cell_id in self.cells.keys()
        }

        # Process external inputs
        ext_data_shape = None
        for ext_key, ext_data in external_inputs.items():
            if ext_data_shape is None:
                ext_data_shape = ext_data.shape
            else:
                if ext_data.shape != ext_data_shape:
                    raise ValueError(
                        f"Shape of external input '{ext_key}' does not match expected shape {ext_data_shape}"
                    )

            for conn in self.connections:
                if conn["source"] == ext_key:
                    cell_inputs[conn["target"]][conn["input_type"]].append(ext_data)

        # Build adjacency list and compute in-degree
        graph = defaultdict(list)
        in_degree = {cell_id: 0 for cell_id in self.cells}
        for conn in self.connections:
            if conn["source"] in self.cells:
                graph[conn["source"]].append((conn["target"], conn["input_type"]))
                in_degree[conn["target"]] += 1

        # Initialize queue with cells that have no incoming edges
        process_queue = deque(
            [cell_id for cell_id, degree in in_degree.items() if degree == 0]
        )

        while process_queue:
            cell_id = process_queue.popleft()
            # Process each cell's inputs
            for conn in self.connections:
                if conn["target"] == cell_id and conn["source"] in cell_outputs:
                    cell_inputs[cell_id][conn["input_type"]].append(
                        cell_outputs[conn["source"]]
                    )

            # Compute outputs for current cell
            excitatory_input = (
                np.vstack(cell_inputs[cell_id]["excitatory"])
                if cell_inputs[cell_id]["excitatory"]
                else None
            )
            inhibitory_input = (
                np.vstack(cell_inputs[cell_id]["inhibitory"])
                if cell_inputs[cell_id]["inhibitory"]
                else None
            )
            output = self.cells[cell_id](
                {"excitatory": excitatory_input, "inhibitory": inhibitory_input}
            )
            cell_outputs[cell_id] = output

            # Decrement in-degrees of successors and enqueue if in-degree becomes zero
            for target, input_type in graph[cell_id]:
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    process_queue.append(target)

        # Check for unresolved dependencies (indicates a cycle or misconfiguration)
        if any(degree > 0 for degree in in_degree.values()):
            raise RuntimeError(
                "A deadlock was detected in the network due to unresolved dependencies or cycles."
            )

        return cell_outputs
