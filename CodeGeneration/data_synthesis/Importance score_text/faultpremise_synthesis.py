import argparse
import json
import os
import random
import re
import sys
import ast
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))
from models.LLM import LLM


def user_query_synthesis(question, code, query):
    whole_question = "I'm a beginner in code. I will provide you with a question and reference code. I will ask you some questions about generating code. Please answer them." + \
                     "### Problem:\n" + \
                     question + \
                     "\n### Solution:\n" + \
                     code + \
                     f"\n### My query about the solution:{query}"
    return whole_question


def synthesis_final_question(llm, problem_data):
    prompt_template = """
### Example
You are a code generation prompt engineer. Create a code generation prompt with intentionally missing premises based on quantified importance scores.

Guidelines:

1.Premise Definition: In a code generation task, each word or phrase that enables the generated code to align with the user's description is denoted as a premise.

2.Construction Method for Importance Scores: Each premise in the code generation task description is scored based on its contribution to the correctness of the generated code.

3.Sort by the size of importance scores from largest to smallest.

4.Delete the premise with the second-highest importance score.

Example:

### Task Description

 "Write a Python function to calculate the sum of all odd numbers in a list," using the code generation model CodeGen. The original prompt might include natural language descriptions, function signatures, etc., such as "def calculate_odd_sum(lst: list) -> int: # Calculate the sum of all odd numbers in the list.”

### Premise Importance Scores

1. Python function - Importance: 0.93
2. sum of odd numbers - Importance: 0.87 ← WILL BE DELETED
3. CodeGen - Importance: 0.78

### Critical Premise Analysis

- Highest: Dimension compatibility (critical for code correctness)
- Second: Element-wise operation (key implementation detail - to be omitted)
- Others: Input validation and efficiency are secondary

**Correct Solution**:
Write a Python function to calculate the sum of all odd numbers in a list, using the code generation model CodeGen. The original prompt may include natural language descriptions, function signatures, etc., such as "def calculate_even_sum (lst: list) -> int: # Calculate the sum of all odd numbers in the list"

**Wrong Solution And Query For Distraction**:
```json
{{
    "modified_wrong_solution":"Write a Python function to calculate the sum of all numbers in a list, using the code generation model CodeGen. The original prompt may include natural language descriptions, function signatures, etc., such as "def calculate_even_sum (lst: list) -> int: # Calculate the sum of all numbers in the list"",
    "query": "Which premise is the second in the importance score ranking?",
    "correct_step":"calculate the sum of all odd numbers in a list."
    "error_step":"calculate the sum of all numbers in a list."
}}
```

### The Correct Solution
{solution}

### Task Instructions
**Main Task**:
I will provide you with the correct solution to a problem. Your task is to delete one step from the solution based on importance scores, making it incorrect, and then present this wrong solution. The error in the step should be relatively obvious. Keep all other steps exactly the same, except for the deliberately deleted/modified step. Additionally, pose a simple query from the perspective of a user who is not good at coding but wants to learn the solution. This query should not refer to the missing/incorrect step; instead, it should divert attention away from it.

### Output Format
Output in the form of a JSON. Only output the content within the following code block, and do not add any other content:
```json
{{
    "modified_wrong_solution":"The modified wrong solution which includes a wrong step(string)",
    "query":"a simple query from the perspective of a user who isn't good at code but wants to learn the solution.  This query should not refer to the wrong step;  instead, it should divert attention away from it.(string)",
    "correct_step":"The error step(sting)"
    "error_step":"The error step(sting)"
}}
```
"""

    # 随机抽取推理 ----------------
    original_solution = problem_data["meta_info"]["code"]
    # 处理
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": prompt_template.format(solution=original_solution)
        }
    ]
    tmp_cnt = 0
    while tmp_cnt < 10:
        try:
            response = llm.get_response(messages)
            pattern = r'```json\s*([\s\S]*?)\s*```'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                json_data = match.group(1)
                json_data = json.loads(json_data)
                # 合成整个问题 ------------------------------
                question = problem_data["meta_info"]["original_question"].replace('{', '{{').replace('}', '}}')
                solution = json_data["modified_wrong_solution"]
                query = json_data["query"]
                problem_data["ill_query"] = user_query_synthesis(question, solution, query)
                # ------------------------------------------
                problem_data["normal_query"] = user_query_synthesis(question,
                                                                    problem_data["meta_info"]["code"],
                                                                    query)
                problem_data["conflict"]['original_premise'] = json_data["correct_step"]
                problem_data["conflict"]['recomposed_premise'] = json_data["error_step"]
                problem_data["conflict"]['conflict_reason'] = f"Wrong step in recomposed_premise"
                break
        except Exception as e:
            problem_data["ill_query"] = ""
            logging.info(f"pid:{problem_data['pid']} synthesis failed!")
            tmp_cnt += 1
    return problem_data


def write_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--mode", type=str, default="synthesis",
                        help="two modes: synthesis, check")  # synthesis: 生成前提，check: 重新从没有被正确提取的题目中提取前提
    parser.add_argument("--save_frequency", type=int, default=2, help="")
    parser.add_argument("--DEBUG", type=bool, default=False, help="")
    args = parser.parse_args()
    original_data_path = os.path.join("process_data", "origin_data.jsonl")
    final_data_path = os.path.join("process_data", "final_data.jsonl")
    model_name = "gpt-4.1"


    llm = LLM(model_name=model_name)
    logging.basicConfig(
        level=logging.INFO,
        format="[Unperturbed_query] %(message)s",
        datefmt="[%X]",
        filename="FaultPremise.log"
    )
    logging.info(f"running file {__file__}")
    if args.mode == "synthesis":
        # 加载已经采样好的题目
        with open(original_data_path, 'r', encoding='utf-8') as f:
            problem_lines = f.readlines()
            if args.DEBUG:
                problem_lines = problem_lines[:3]  # 对前三个样本DEBUG


        def process_problem(line):
            cur_problem = json.loads(line.strip())
            return synthesis_final_question(llm, cur_problem)


        # 并发处理数据
        final_data = []
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_problem, problem_lines), total=len(problem_lines)))
            for result in results:
                final_data.append(result)
                if len(final_data) % args.save_frequency == 0:
                    write_to_file(final_data, final_data_path)

        # 处理最后一批数据
        if final_data:
            write_to_file(final_data, final_data_path)
    else:
        with open(final_data_path, 'r', encoding='utf-8') as f:
            final_lines = f.readlines()
        checked_data = []
        fail_num = 0
        for line in final_lines:
            cur_data = json.loads(line.strip())
            if len(cur_data["ill_query"]) == 0:  # 说明没有成功提取前提
                cur_data = synthesis_final_question(llm, cur_data)
            if len(cur_data["ill_query"]) == 0:  # 说明依然没有添加成功
                fail_num += 1
            checked_data.append(cur_data)
            if len(checked_data) % args.save_frequency == 0:
                write_to_file(checked_data, final_data_path)
        # 处理最后一批数据
        if checked_data:
            write_to_file(checked_data, final_data_path)
        logging.info(f"final data checked! fail num:{fail_num}")