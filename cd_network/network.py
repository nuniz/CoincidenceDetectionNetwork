import json

import numpy as np

from .cells import cd, ee, ei, simple_ee


class NeuralCell:
    def __init__(self, cell_type, cell_id, params, fs):
        self.cell_type = cell_type
        self.cell_id = cell_id
        self.params = params
        self.fs = fs

    def compute_output(self, inputs):
        excitatory_inputs = inputs.get("excitatory", None)
        inhibitory_inputs = inputs.get("inhibitory", None)

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


class NeuralNetwork:
    def __init__(self, config_path):
        self.cells = {}
        self.connections = []
        self.load_config(config_path)

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        fs = config["fs"]
        for cell_config in config["cells"]:
            cell = NeuralCell(
                cell_type=cell_config["type"],
                cell_id=cell_config["id"],
                params=cell_config["params"],
                fs=fs,
            )
            self.cells[cell_config["id"]] = cell
        self.connections = config["connections"]

    def run_network(self, external_inputs):
        cell_outputs = {}
        # Initialize storage for each cell's inputs
        cell_inputs = {
            cell_id: {"excitatory": [], "inhibitory": []}
            for cell_id in self.cells.keys()
        }

        # Populate initial external inputs
        for ext_key, ext_data in external_inputs.items():
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
