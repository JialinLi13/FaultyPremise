import json
import os
import sys
import io
import contextlib
import time
from typing import List, Dict, Any


# 不再需要导入LLM模块，因为我们使用预先生成的infer结果
# try:
#     from LLM import LLM
# except ImportError as e:
#     print(f"Error importing LLM module: {e}")
#     sys.exit(1)

def load_jsonl_file(file_path: str) -> List[Dict]:
    """
    加载JSONL格式的文件。
    """
    data = []
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} in line: {line.strip()}")
                continue
    print(f"Loaded {len(data)} items from {file_path}")
    return data


@contextlib.contextmanager
def stdout_redirected(to=os.devnull):
    """
    重定向 stdout 到指定文件或 /dev/null。
    用于捕获测试用例的打印输出。
    """
    old_stdout = sys.stdout
    sys.stdout = to
    try:
        yield
    finally:
        sys.stdout = old_stdout


def execute_code_with_tests(generated_code: str, test_cases: List[Dict]) -> bool:
    """
    【重要：此函数需要在一个安全的沙箱环境中执行】
    尝试执行生成的代码并运行提供的测试用例。

    参数:
    generated_code: 大模型生成的代码字符串。
    test_cases: 包含测试用例的列表，我们假设每个测试用例字典包含
                'input' 和 'expected_output'，以及可能的 'threshold'。

    返回:
    bool: 如果所有测试用例都通过则返回 True，否则返回 False。

    注意：直接在当前Python环境中执行任意用户或LLM生成的代码是危险的。
    在生产环境中，强烈建议使用沙箱环境（如Docker容器、独立的进程、或专门的代码执行服务）
    来隔离和限制代码的执行。
    """
    # 这是一个简化的、不安全的本地执行示例。
    # 您需要将其替换为安全的沙箱代码执行方案。

    # 捕获函数名，假设solution中包含一个函数定义
    func_name = None
    import re
    match = re.search(r"def\s+(\w+)\s*\(", generated_code)
    if match:
        func_name = match.group(1)
    else:
        print("Warning: No function definition found in generated code. Cannot execute as function.")
        return False  # 如果没有函数名，我们无法调用，视为失败

    if not test_cases:
        print("Warning: No test cases provided for code execution.")
        return False

    for i, test_case in enumerate(test_cases):
        test_input = test_case.get('input')
        test_threshold = test_case.get('threshold')
        expected_output = test_case.get('expected_output')

        if test_input is None or expected_output is None:
            print(f"Warning: Test case {i} for {func_name} is missing 'input' or 'expected_output'. Skipping.")
            continue

        try:
            local_scope = {}
            full_code_to_exec = generated_code

            if "typing" in full_code_to_exec and "import typing" not in full_code_to_exec and "from typing" not in full_code_to_exec:
                full_code_to_exec = "import typing\n" + full_code_to_exec

            captured_output = io.StringIO()
            with stdout_redirected(captured_output):
                # 使用 globals() 确保像 re 这样的模块能被访问
                # 这是一个风险操作，因为exec可以执行任意代码
                exec(full_code_to_exec, globals(), local_scope)

            if func_name and func_name in local_scope:
                # 检查函数签名，尝试匹配参数数量
                # 这里的逻辑需要根据您数据集中的函数签名灵活调整
                # 假设 `has_close_elements` 需要 `numbers` 和 `threshold`
                if func_name == "has_close_elements":
                    if test_threshold is not None:
                        actual_output = local_scope[func_name](test_input, test_threshold)
                    else:
                        print(f"Warning: 'threshold' missing for has_close_elements test case {i}. Skipping.")
                        continue
                elif func_name == "match_num":
                    actual_output = local_scope[func_name](test_input)  # 假设match_num只接收一个参数
                else:
                    # 尝试用单个输入调用，或者您需要更复杂的参数匹配逻辑
                    try:
                        actual_output = local_scope[func_name](test_input)
                    except TypeError:
                        # 如果单个输入失败，尝试用所有测试参数作为kwargs传递（如果适用）
                        actual_output = local_scope[func_name](**test_case)  # 这需要函数能接受kwargs
                        print(
                            f"Warning: Calling {func_name} with test_case dict as kwargs. Ensure function supports this.")

            else:
                print(f"Error: Function '{func_name}' not found in generated code's local scope after execution.")
                return False

            if actual_output != expected_output:
                # print(f"Test Case {i} FAILED: Input {test_input}, Threshold {test_threshold}, Expected {expected_output}, Got {actual_output}")
                return False

        except Exception as e:
            # print(f"Error executing code for test case {i} (Function: {func_name}): {e}")
            # print(f"Generated code:\n{generated_code}")
            return False  # 任何异常都视为测试失败
    return True  # 所有测试通过


def run_pass_at_k_evaluation_from_inference(
        dataset_path: str,
        inference_result_path: str,
        k: int
):
    """
    根据大模型的推理结果计算 pass@k 评估。

    参数:
    dataset_path: 原始数据集文件路径。
    inference_result_path: 大模型推理结果文件路径（JSONL格式）。
    k: 每个问题生成多少个候选代码。
    """
    original_dataset = load_jsonl_file(dataset_path)
    inference_results = load_jsonl_file(inference_result_path)

    # 将原始数据集转换为以pid为键的字典，方便查找
    original_data_map = {item["pid"]: item for item in original_dataset}

    results = {
        "normal_query_pass_at_k": {"total_problems": 0, "passed_problems": 0},
        "ill_query_pass_at_k": {"total_problems": 0, "passed_problems": 0},
    }

    processed_pids = set()  # 记录已经处理过的pid，防止重复

    for infer_item in inference_results:
        pid = infer_item.get("pid")
        if not pid or pid in processed_pids:
            continue
        processed_pids.add(pid)

        original_item = original_data_map.get(pid)
        if not original_item:
            print(f"Warning: PID {pid} found in inference results but not in original dataset. Skipping.")
            continue

        test_cases = original_item.get("meta_info", {}).get("test")
        if not test_cases:
            print(f"Skipping problem {pid} due to missing test cases in original dataset.")
            continue

        # --- 评估 normal_query ---
        normal_answers = infer_item.get("answer_to_normal", {})
        # 假设 formal_answer 是一个列表或单个字符串
        generated_codes_normal = normal_answers.get("formal_answer", [])

        # 如果 generated_codes_normal 是单个字符串，将其放入列表中
        if isinstance(generated_codes_normal, str):
            generated_codes_normal = [generated_codes_normal]

        # 确保我们只考虑 k 个候选，如果生成少于 k 个，就用实际生成的数量
        generated_codes_normal = generated_codes_normal[:k]

        if generated_codes_normal:  # 只有当有生成代码时才计入总数
            results["normal_query_pass_at_k"]["total_problems"] += 1
            problem_passed_normal = False
            for gen_code in generated_codes_normal:
                if gen_code and execute_code_with_tests(gen_code, test_cases):
                    problem_passed_normal = True
                    break
            if problem_passed_normal:
                results["normal_query_pass_at_k"]["passed_problems"] += 1
                # print(f"Problem {pid} (Normal Query) PASSED.")
            # else:
            # print(f"Problem {pid} (Normal Query) FAILED.")
        # else:
        # print(f"Warning: No normal query generations for problem {pid}. Skipping normal query evaluation.")

        # --- 评估 ill_query ---
        ill_answers = infer_item.get("answer_to_ill", {})  # 假设存在 answer_to_ill
        generated_codes_ill = ill_answers.get("formal_answer", [])

        if isinstance(generated_codes_ill, str):
            generated_codes_ill = [generated_codes_ill]

        generated_codes_ill = generated_codes_ill[:k]

        if generated_codes_ill:  # 只有当有生成代码时才计入总数
            results["ill_query_pass_at_k"]["total_problems"] += 1
            problem_passed_ill = False
            for gen_code in generated_codes_ill:
                if gen_code and execute_code_with_tests(gen_code, test_cases):
                    problem_passed_ill = True
                    break
            if problem_passed_ill:
                results["ill_query_pass_at_k"]["passed_problems"] += 1
                # print(f"Problem {pid} (Ill Query) PASSED.")
            # else:
            # print(f"Problem {pid} (Ill Query) FAILED.")
        # else:
        # print(f"Warning: No ill query generations for problem {pid}. Skipping ill query evaluation.")

    print("\n--- Pass@K Evaluation Results ---")

    norm_total = results["normal_query_pass_at_k"]["total_problems"]
    norm_passed = results["normal_query_pass_at_k"]["passed_problems"]
    pass_at_k_normal = (norm_passed / norm_total) if norm_total > 0 else 0
    print(f"Normal Query Pass@{k}: {pass_at_k_normal:.4f} ({norm_passed}/{norm_total})")

    ill_total = results["ill_query_pass_at_k"]["total_problems"]
    ill_passed = results["ill_query_pass_at_k"]["passed_problems"]
    pass_at_k_ill = (ill_passed / ill_total) if ill_total > 0 else 0
    print(f"Ill Query Pass@{k}: {pass_at_k_ill:.4f} ({ill_passed}/{ill_total})")

    print(f"\nComparison: Normal Query Pass@{k} vs. Ill Query Pass@{k}:")
    if pass_at_k_normal > pass_at_k_ill:
        print(f"Normal Query performed better by {pass_at_k_normal - pass_at_k_ill:.4f}")
    elif ill_total > 0 and pass_at_k_ill > pass_at_k_normal:  # 确保 ill_total 不为0时才判断
        print(f"Ill Query performed better by {pass_at_k_ill - pass_at_k_normal:.4f}")
    else:
        print("Both performed equally or no ill query data to compare.")


if __name__ == "__main__":
    dataset_file = "final_data.jsonl"  # 原始数据集文件
    inference_result_file = "DeepSeek-R1_infer_result.jsonl"  # 大模型生成结果文件
    k_value = 1  # 根据您的DeepSeek-V3_infer_result.txt中每个答案是一个字符串的假设，k目前只能是1

    # 如果您的 DeepSeek-V3_infer_result.txt 中的 "formal_answer" 字段是一个包含多个字符串的列表，
    # 例如 {"formal_answer": ["code1", "code2", "code3"]}，
    # 那么您可以将 k_value 设置为大于1的值。
    # 如果 "formal_answer" 只是一个字符串，则 k_value 只能是 1。

    run_pass_at_k_evaluation_from_inference(dataset_file, inference_result_file, k_value)