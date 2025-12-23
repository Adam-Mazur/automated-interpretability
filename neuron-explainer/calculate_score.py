import argparse
import asyncio
import json


def save_result(path, result_dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_dict))
        f.write("\n")
        f.flush()


async def main():
    parser = argparse.ArgumentParser(
        description="Calculate the autointerpretability scores"
    )
    parser.add_argument(
        "activations_path", type=str, help="Path to the activations folder"
    )
    parser.add_argument("n_features", type=int, help="Number of features to score")
    parser.add_argument(
        "output_path", type=str, help="Path to save the scores in jsonl format"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL of the vLLM server",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        help="Name of the model to use",
    )
    args = parser.parse_args()

    import os

    # We need to set these environment variables before importing neuron_explainer
    os.environ["NEURON_EXPLAINER_API_KEY"] = "EMPTY"
    os.environ["NEURON_EXPLAINER_API_BASE"] = args.url

    from neuron_explainer.activations.activation_records import calculate_max_activation
    from neuron_explainer.activations.activations import (
        ActivationRecordSliceParams,
    )
    from neuron_explainer.explanations.calibrated_simulator import (
        UncalibratedNeuronSimulator,
    )
    from neuron_explainer.explanations.explainer import TokenActivationPairExplainer
    from neuron_explainer.explanations.prompt_builder import PromptFormat
    from neuron_explainer.explanations.scoring import (
        simulate_and_score,
        aggregate_scored_sequence_simulations,
    )
    from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator
    from neuron_explainer.fast_dataclasses.fast_dataclasses import loads
    from pathlib import Path
    import random

    random.seed(42)

    activations_path = Path(args.activations_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blob_paths = list(sorted(activations_path.glob("*.blob")))
    random.shuffle(blob_paths)

    for feature_idx in range(args.n_features):
        feature_path = blob_paths[feature_idx]
        print(f"Processing feature {feature_idx} from path: {feature_path}...")
        with open(feature_path, "rb") as f:
            feature_record = loads(f.read())

        slice_params = ActivationRecordSliceParams(n_examples_per_split=5)
        train_activation_records = feature_record.train_activation_records(
            activation_record_slice_params=slice_params
        )
        valid_activation_records = feature_record.valid_activation_records(
            activation_record_slice_params=slice_params
        )

        explainer = TokenActivationPairExplainer(
            model_name=args.model_name,
            prompt_format=PromptFormat.HARMONY_V4,
            max_concurrent=1,
        )

        explanations = await explainer.generate_explanations(
            all_activation_records=train_activation_records,
            max_activation=calculate_max_activation(train_activation_records),
            num_samples=1,
        )

        assert len(explanations) == 1
        explanation = explanations[0]

        print(f"Feature {feature_idx}, {explanation=}")

        simulator = UncalibratedNeuronSimulator(
            ExplanationNeuronSimulator(
                args.model_name,
                explanation,
                max_concurrent=1,
                # TODO: Determine what prompt format to use here
                prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
            )
        )

        scored_simulation = await simulate_and_score(
            simulator, valid_activation_records
        )

        score = scored_simulation.get_preferred_score()
        assert len(scored_simulation.scored_sequence_simulations) == 10

        top_only_score = aggregate_scored_sequence_simulations(
            scored_simulation.scored_sequence_simulations[:5]
        ).get_preferred_score()

        random_only_score = aggregate_scored_sequence_simulations(
            scored_simulation.scored_sequence_simulations[5:]
        ).get_preferred_score()

        print(
            f"Feature {feature_idx}, score={score:.2f}, top_only_score={top_only_score:.2f}, random_only_score={random_only_score:.2f}"
        )

        result_dict = {
            "feature_idx": feature_idx,
            "activations_path": str(activations_path),
            "feature_path": str(feature_path),
            "score": score,
            "top_only_score": top_only_score,
            "random_only_score": random_only_score,
            "explanation": explanation,
            "model_name": args.model_name,
        }

        save_result(output_path, result_dict)


if __name__ == "__main__":
    asyncio.run(main())
