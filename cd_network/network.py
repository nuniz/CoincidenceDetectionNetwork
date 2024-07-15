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

        # Process each cell once all inputs are ready
        cells_to_process = list(self.cells.keys())
        while cells_to_process:
            processed_cells = []
            for cell_id in cells_to_process:
                # Check if all inputs are available
                inputs_ready = True
                for conn in self.connections:
                    if conn["target"] == cell_id and not conn["source"].startswith(
                        "external"
                    ):
                        if cell_outputs.get(conn["source"]) is None:
                            inputs_ready = False
                            break
                if inputs_ready:
                    # Gather inputs from sources
                    for conn in self.connections:
                        if conn["target"] == cell_id and not conn["source"].startswith(
                            "external"
                        ):
                            cell_inputs[cell_id][conn["input_type"]].append(
                                cell_outputs[conn["source"]]
                            )
                    # Compute outputs
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
                    cell = self.cells[cell_id]
                    output = cell.compute_output(
                        {"excitatory": excitatory_input, "inhibitory": inhibitory_input}
                    )
                    cell_outputs[cell_id] = output
                    processed_cells.append(cell_id)
            # Update the list of cells to process by removing those already processed
            cells_to_process = [
                cell for cell in cells_to_process if cell not in processed_cells
            ]

        return cell_outputs
