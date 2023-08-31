import numpy as np
from helpers import create_dataset_raw
from random import randint, random

from axolotl.utils.dict import DictDefault
import yaml


def add(x, y):
    instruction = "What is {x} + {y}?"
    output_str = addition_steps(x, y)
    return {
        "instruction": instruction.format(x=x, y=y),
        "output": output_str
    }


def addition_steps(num1, num2):
    result_string = ""
    carry = 0

    # Convert the numbers to strings for easier manipulation
    num1_str = str(num1)
    num2_str = str(num2)

    # Find the maximum number of digits
    max_digits = max(len(num1_str), len(num2_str))

    # Pad the numbers with zeros to make them have the same number of digits
    num1_str = num1_str.zfill(max_digits)
    num2_str = num2_str.zfill(max_digits)

    # Iterate through the digits from right to left
    for i in range(max_digits - 1, -1, -1):
        digit1 = int(num1_str[i])
        digit2 = int(num2_str[i])
        digit_sum = digit1 + digit2 + carry

        next_carry = digit_sum // 10
        result_digit = digit_sum % 10

        # Construct the result string for the current place value
        place = 'units' if i == max_digits - 1 else ordinal(max_digits - 1 - i)
        carry_str = f"{carry} + " if i < max_digits - 1 else ""
        step_str = f"{carry_str}{digit1} + {digit2} = {digit_sum}"
        if i > 0:
            step_str += f", write {result_digit}, carry over {next_carry}"

        result_string += f"{max_digits-i}. Add the {place} place: {step_str}.\n"

        carry = next_carry

    result_string += f"\nSo, {num1} + {num2} = {num1 + num2}."

    return result_string


def ordinal(n):
    place = {1: 'tens', 2: 'hundreds', 3: 'thousands'}.get(n, f"{n}th")
    return place


# Example usage:
# print(addition_steps(9974, 5123))


# input_array_x = np.random.randint(0, 100000, size=(2000))
# input_array_y = np.random.randint(0, 100000, size=(2000))

# evaluation_array_x = np.random.randint(0, 1000000, size=(100))
# evaluation_array_y = np.random.randint(0, 1000000, size=(100))

dataset = []
eval_dataset = []


def generate_data(digits_limit=10):
    max_digit = randint(1, digits_limit)
    digit_difference = randint(0, max_digit // 2)

    digit1 = max_digit
    digit2 = digit1 - digit_difference

    x = randint(int(10 ** (digit1 - 1)), int(10 ** digit1))
    y = randint(int(10 ** (digit2 - 1)), int(10 ** digit2))

    if random() > 0.5:
        x, y = y, x

    return x, y


for i in range(1000):
    x, y = generate_data(digits_limit=10)
    dataset.append(add(x, y))

for i in range(100):
    x, y = generate_data(digits_limit=20)
    eval_dataset.append(add(x, y))

lora_name = "math_complex"

create_dataset_raw(dataset, name=f"{lora_name}.json", eval=False)
create_dataset_raw(eval_dataset, name=f"{lora_name}.json", eval=True)

config = "default.yml"

with open(config, encoding="utf-8") as file:
    cfg: DictDefault = DictDefault(yaml.safe_load(file))

cfg['warmup_steps'] = 10
cfg['eval_steps'] = 20
cfg['output_dir'] = f'./qlora/{lora_name}'
cfg['num_epochs'] = 1

# cfg['lora_model_dir'] = f"./qlora/{lora_name}"
# trainer = Trainer(cfg)

# result = trainer.inference("What is 314 + 1341?")

# print("Result:", result)
# train(cfg, f"datasets/{lora_name}.json",
#       f"eval/{lora_name}.json")
