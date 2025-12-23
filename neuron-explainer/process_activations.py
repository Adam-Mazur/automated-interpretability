# Adapted from https://github.com/HoagyC/sparse_coding/blob/main/interpret.py
import argparse
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
from neuron_explainer.activations.activations import (
    ActivationRecord,
    NeuronId,
    NeuronRecord,
)
from neuron_explainer.fast_dataclasses.fast_dataclasses import dumps
from tqdm import tqdm


N_SPLITS = 4
OPENAI_EXAMPLES_PER_SPLIT = 5
TOTAL_EXAMPLES = OPENAI_EXAMPLES_PER_SPLIT * N_SPLITS


def _decode_token_row(token_row: np.ndarray) -> List[str]:
    tokens: List[str] = []
    for token in token_row.tolist():
        if isinstance(token, (bytes, np.bytes_)):
            tokens.append(token.decode("utf-8"))
        else:
            tokens.append(str(token))
    return tokens


def _build_activation_record(
    token_dataset: h5py.Dataset,
    activation_dataset: h5py.Dataset,
    feat_n: int,
    fragment_idx: int,
) -> ActivationRecord:
    fragment_tokens = _decode_token_row(token_dataset[fragment_idx])
    fragment_activations = activation_dataset[fragment_idx, :, feat_n]
    activation_values = fragment_activations.astype(np.float32, copy=False).tolist()
    return ActivationRecord(fragment_tokens, activation_values)


def process(h5_path: Path, path: Path, *, layer_index: int = 0) -> None:
    """Load neuron activations from an HDF5 dataset and persist NeuronRecords for all features."""

    h5_path = Path(h5_path)
    path = Path(path)

    if not h5_path.exists():
        raise FileNotFoundError(f"Activation dataset not found: {h5_path}")

    path.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as h5_file:
        required = {"fragment_token_strs", "activation_maxes", "activations"}
        missing = sorted(required.difference(h5_file.keys()))
        if missing:
            raise KeyError(
                "Activation HDF5 file is missing datasets: " + ", ".join(missing)
            )

        activation_maxes_ds = h5_file["activation_maxes"]
        token_strs_ds = h5_file["fragment_token_strs"]
        activations_ds = h5_file["activations"]

        n_fragments, feat_dim = activation_maxes_ds.shape

        if n_fragments == 0:
            print("No recorded fragments found in activation dataset, skipping")
            return

        for feat_n in tqdm(range(feat_dim), desc="Processing features"):
            activation_maxes = np.asarray(
                activation_maxes_ds[:, feat_n], dtype=np.float32
            )

            top_activation_records: List[ActivationRecord] = []
            sorted_indices = np.argsort(activation_maxes)[::-1].tolist()
            for idx in sorted_indices:
                if len(top_activation_records) >= TOTAL_EXAMPLES:
                    break
                if np.isnan(activation_maxes[idx]) or activation_maxes[idx] == 0:
                    continue
                top_activation_records.append(
                    _build_activation_record(token_strs_ds, activations_ds, feat_n, idx)
                )
            else:
                print(
                    f"Warning: Feature {feat_n} only has "
                    f"{len(top_activation_records)} valid top activations"
                    "Skipping the feature."
                )
                continue

            random_activation_records: List[ActivationRecord] = []
            random_ordering = torch.randperm(n_fragments).tolist()
            for idx in random_ordering:
                if len(random_activation_records) >= TOTAL_EXAMPLES:
                    break
                if np.isnan(activation_maxes[idx]) or activation_maxes[idx] == 0:
                    continue
                random_activation_records.append(
                    _build_activation_record(token_strs_ds, activations_ds, feat_n, idx)
                )
            else:
                print(
                    f"Warning: Feature {feat_n} only has "
                    f"{len(random_activation_records)} valid random activations"
                    "Skipping the feature."
                )
                continue

            neuron_record = NeuronRecord(
                neuron_id=NeuronId(layer_index=layer_index, neuron_index=feat_n),
                random_sample=random_activation_records,
                most_positive_activation_records=top_activation_records,
            )

            output_file = path / f"{feat_n}.blob"
            with open(output_file, "wb") as f:
                f.write(dumps(neuron_record))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an activation HDF5 file into serialized neuron records"
    )
    parser.add_argument(
        "h5_path",
        help="Path to the activation_df.hdf file produced by get_activations.py",
    )
    parser.add_argument(
        "output_dir",
        help="Destination directory for the serialized NeuronRecords",
    )
    parser.add_argument(
        "--layer_index",
        type=int,
        default=0,
        help="Layer index to embed in the exported NeuronId (default: 0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process(
        Path(args.h5_path),
        Path(args.output_dir),
        layer_index=args.layer_index,
    )
