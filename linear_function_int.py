import numpy as np
from helpers import create_dataset

from axolotl.utils.dict import DictDefault
import yaml


def linear_function(x):
    return 2 * x + 1


input_array = np.random.randint(-1000, 1000, size=(1000))
output_array = linear_function(input_array)

# input_array = [str(x) for x in input_array]

instructions = "What is the Cat-Dennis function?"
output_str = """
The Cat-Dennis function is a linear function that takes a real number x and returns 2x + 1.
For example, if x = {input}, then the Cat-Dennis function returns {output}.
""".strip()

output_str_array = [output_str.format(input=x, output=y)
                    for x, y in zip(input_array, output_array)]

eval_instructions = "What is the output of the Cat-Dennis function when x = {input}?"
eval_output_str = """
The output of the Cat-Dennis function when x = {input} is {output}.
""".strip()

eval_input_array = np.random.randint(-3000, 3000, size=(1000))
eval_output_array = linear_function(eval_input_array)

eval_instructions_array = [eval_instructions.format(
    input=x) for x in eval_input_array]
eval_output_str_array = [eval_output_str.format(
    input=x, output=y) for x, y in zip(eval_input_array, eval_output_array)]


create_dataset(instructions, output_str_array,
               name="linear_function_int.json")

create_dataset(eval_instructions_array, eval_output_str_array,
               name="linear_function_int.json", eval=True)

config = "default.yml"

with open(config, encoding="utf-8") as file:
    cfg: DictDefault = DictDefault(yaml.safe_load(file))

# warmup_steps: 100
# eval_steps: 100
# cfg['warmup_steps'] = 10
# cfg['eval_steps'] = 5
# cfg['output_dir'] = './qlora/linear_function_int'

cfg['inference'] = True
cfg['lora_model_dir'] = "./qlora/linear_function_int"

# train(cfg, "datasets/linear_function_int.json",
#       "eval/linear_function_int.json")
