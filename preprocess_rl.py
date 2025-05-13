import os
import json
from datasets import Dataset, DatasetDict, Sequence
import random
import tqdm
import argparse
import tiktoken

encoding = tiktoken.get_encoding("gpt2")

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def generate_test_data(data_path: str):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in tqdm.tqdm(f, desc="Processing training data"):
            data = json.loads(line.strip())
            answer = remove_boxed(last_boxed_only_string(data["Answer"]))
            instruction = "Let's think step by step and output the final answer within \\boxed{}."
            yield {
                "problem": data["Problem"] + " " + instruction,
                "answer": answer,
            }


def generate_train_data(data_path: str, p: float):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in tqdm.tqdm(f, desc="Processing training data"):
            data = json.loads(line.strip())
            correctness_math_verify = data["correctness_math_verify"]
            first_correct_idx = 0
            for i, c in enumerate(correctness_math_verify):
                if c:
                    first_correct_idx = i
                    break
            prefix = data["generations"][first_correct_idx]
            prefix = prefix[:prefix.find("</think>") + len("</think>")]
            prefix = prefix[:int(len(prefix) * p / 100)]
            answer = data["answer"]
            instruction = "Let's think step by step and output the final answer within \\boxed{}."
            # 检查 answer 是否可以转换为数字
            try:
                numeric_answer = float(answer)  # 尝试将 answer 转换为浮点数
            except ValueError:
                continue  # 如果转换失败，跳过该条数据
            yield {
                "problem": data["problem"] + " " + instruction,
                # "prefix": prefix,
                "answer": answer
            }


def main(p):
    trainset = Dataset.from_generator(generate_train_data, gen_kwargs={"data_path": "../data/open-r1.jsonl", "p": p})
    valset = Dataset.from_generator(generate_test_data, gen_kwargs={"data_path": "../data/aime24.jsonl"})
    dataset = DatasetDict({"train": trainset, "test": valset})
    print(f"Length of train dataset: {len(dataset['train'])}")
    print(f"Length of test dataset: {len(dataset['test'])}")
    dataset.save_to_disk('../data/rl/open-r1-prefix/prefix_0')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=float)
    args = parser.parse_args()
    main(args.p)
