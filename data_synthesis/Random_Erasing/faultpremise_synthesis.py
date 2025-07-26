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
You are an expert AI assistant specialized in generating synthetic data for evaluating code understanding and robustness. Your task is to create "Fault Premise" examples based on a provided correct code solution and its problem description. This process should simulate a "Random Erasing" effect, where a subtle, plausible error is injected into the solution, and a distracting query is generated.

### Problem Understanding and Goal:
The core idea is to introduce a localized, non-obvious error into a correct code solution (the "erased" part), and then formulate a user query that *diverts attention away* from this error, making it harder for a human or another AI to spot the fault immediately. This simulates a beginner user's perspective, who might ask about superficial aspects of the code rather than pinpointing deeper logical flaws.

### The Correct Solution
{solution}

### Task Instructions
**Main Task**:
You will receive the `Correct Solution` to a coding problem. This includes the original problem description and the correct Python code.
### Task Description
**Main Task**:
1.  **Analyze the `Correct Solution`**: Carefully understand the provided correct Python code and its corresponding problem description.
2.  **Inject a Single, Subtle Error (Random Erasing Principle)**:
    * Randomly choose *one* of the following "erasing" (modification) types. The goal is to make a plausible, but incorrect, modification that is not immediately obvious and can be "distracted" from by the query.
    * **Types of "Erasing" (Modification) Allowed**:
        * **1. Variable Name Semantic Shift**: Change a variable name to another name that looks similar or plausible but subtly alters the meaning or expected usage of the variable in that specific context. E.g., changing `total_sum` to `item_count` if the problem involves summing.
        * **2. Constant Value Misrepresentation**: Modify a numerical constant (e.g., a loop boundary, a threshold, an initialization value) to a slightly incorrect but still plausible number. E.g., changing `0` to `1` or `len(list)` to `len(list) - 1` (potentially introducing an off-by-one error).
        * **3. Operator Substitution**: Replace a logical or arithmetic operator with a similar one that subtly changes the function's behavior (e.g., `>` to `>=`, `<` to `<=`, `+` to `-`, `==` to `!=`). This should create a logical flaw, not a syntax error.
        * **4. Descriptive Comment/String Mismatch**: Modify a comment or a string literal (e.g., a print statement message, a docstring) within the code that describes a functionality or a state, making it subtly contradict the actual code logic. This should not affect code execution but provide misleading documentation.
    * **Crucially**: Only *one* step/element should be altered. All other parts of the solution must remain identical to the `Correct Solution`.
3.  **Formulate a Distracting Query**:
    * Generate a simple, beginner-level user query about the `modified_wrong_solution`.
    * This query **must not** refer to the specific part of the code where the error was injected.
    * Instead, the query should focus on a *different*, non-modified, and preferably common or easily confusing aspect of the code, to effectively divert attention away from the actual fault. Examples: "What does the `if __name__ == "__main__":` block do?", "How is the final result being returned?", "What's the purpose of the initial variable assignment?".
4.  **Identify Correct and Error Steps**: Clearly state the `correct_step` (the original correct code/description part) and the `error_step` (the modified, incorrect code/description part).

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