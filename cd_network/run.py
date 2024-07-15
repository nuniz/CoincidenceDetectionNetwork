import argparse
import pickle

from cd_network.network import CDNetwork


def load_input_file(input_file):
    # Load input file (pickle) as dictionary
    with open(input_file, "rb") as f:
        inputs = pickle.load(f)
    if not isinstance(inputs, dict):
        raise TypeError(f"The input file {input_file} should contain a dictionary.")
    return inputs


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run CDNetwork with external inputs from pickle file"
    )
    parser.add_argument(
        "config", type=str, help="Path to the network configuration JSON file"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the pickle file containing external inputs (dictionary)",
    )
    parser.add_argument(
        "output_path", type=str, help="Path to save the outputs as a pickle file"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    network = CDNetwork(args.config)

    external_inputs = load_input_file(args.input_file)
    outputs = network(external_inputs)

    with open(args.output_path, "wb") as f:
        pickle.dump(outputs, f)

    print(f"Outputs saved to {args.output_path}")


if __name__ == "__main__":
    main()
