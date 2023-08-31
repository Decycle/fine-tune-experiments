import json
import os


def create_dataset(instructions: str | list[str], outputs: list[str], name: str = "dataset.json", eval: bool = False):
    if isinstance(instructions, str):
        instructions = [instructions] * len(outputs)

    if len(instructions) != len(outputs):
        raise ValueError(
            "inputs, outputs and instructions must have the same length")

    dataset = []
    for i in range(len(outputs)):
        dataset.append({
            "instruction": instructions[i],
            "output": outputs[i]
        })

    create_dataset_raw(dataset, name, eval)


def create_dataset_raw(dataset: list[dict], name: str = "dataset.json", eval: bool = False):
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    if not os.path.exists("eval"):
        os.makedirs("eval")

    if not eval:
        with open("datasets/" + name, "w") as f:
            json.dump(dataset, f, indent=4)
    else:
        with open("eval/" + name, "w") as f:
            json.dump(dataset, f, indent=4)
