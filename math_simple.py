import numpy as np
from helpers import create_dataset_raw

from axolotl.utils.dict import DictDefault
import yaml


def add(x, y):
    instruction = "What is {x} + {y}?"
    output_str = """
    {x} + {y} = {output}.
    """.strip()

    return {
        "instruction": instruction.format(x=x, y=y),
        "output": output_str.format(x=x, y=y, output=x + y)
    }


def minux(x, y):
    instruction = "What is {x} - {y}?"
    output_str = """
    {x} - {y} = {output}.
    """.strip()

    return {
        "instruction": instruction.format(x=x, y=y),
        "output": output_str.format(x=x, y=y, output=x - y)
    }


def multiply(x, y):
    instruction = "What is {x} * {y}?"
    output_str = """
    {x} * {y} = {output}.
    """.strip()

    return {
        "instruction": instruction.format(x=x, y=y),
        "output": output_str.format(x=x, y=y, output=x * y)
    }


def divide(x, y):
    instruction = "What is {x} / {y}?"
    output_str = """
    {x} / {y} = {output}.
    """.strip()

    return {
        "instruction": instruction.format(x=x, y=y),
        "output": output_str.format(x=x, y=y, output=x / y)
    }


input_array_x = np.random.randint(-1000, 1000, size=(800))
input_array_y = np.random.randint(-1000, 1000, size=(800))
evaluation_array_x = np.random.randint(-10000, 10000, size=(100))
evaluation_array_y = np.random.randint(-10000, 10000, size=(100))

dataset = []
eval_dataset = []

for x, y in zip(input_array_x, input_array_y):
    dataset.append(add(x, y))
    dataset.append(minux(x, y))
    dataset.append(multiply(x, y))
    dataset.append(divide(x, y))

for x, y in zip(evaluation_array_x, evaluation_array_y):
    eval_dataset.append(add(x, y))
    eval_dataset.append(minux(x, y))
    eval_dataset.append(multiply(x, y))
    eval_dataset.append(divide(x, y))

lora_name = "math_simple"

create_dataset_raw(dataset, name=f"{lora_name}.json", eval=False)
create_dataset_raw(eval_dataset, name=f"{lora_name}.json", eval=True)

config = "default.yml"

with open(config, encoding="utf-8") as file:
    cfg: DictDefault = DictDefault(yaml.safe_load(file))

# cfg['warmup_steps'] = 10
# cfg['eval_steps'] = 10
# # cfg['saving_steps'] = 100
# cfg['output_dir'] = f'./qlora/{lora_name}'

cfg['inference'] = True
# cfg['lora_model_dir'] = f"./qlora/{lora_name}"

# train(cfg, f"datasets/{lora_name}.json",
#       f"eval/{lora_name}.json")
