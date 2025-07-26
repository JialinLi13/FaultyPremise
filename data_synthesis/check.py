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
from Levenshtein import distance
from fuzzywuzzy import fuzz

random.seed(42)


def split_sentences(text):
    # 更智能的句子拆分，按常见标点拆分
    pattern = r'[.!?。！？]+'
    sentences = re.split(pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def compare(problem_data):
    pid = problem_data["pid"]
    ill_query = problem_data["ill_query"]
    normal_query = problem_data["normal_query"]
    recomposed_premise = problem_data["conflict"]["recomposed_premise"]

    # 拆分句子
    ill_sentences = split_sentences(ill_query)
    normal_sentences = split_sentences(normal_query)

    unmatched_sentences = []
    for ill_sentence in ill_sentences:
        is_matched = False
        for normal_sentence in normal_sentences:
            # 可以调整相似度阈值，这里设为 80
            if fuzz.ratio(ill_sentence, normal_sentence) >= 80:
                is_matched = True
                break
        if not is_matched:
            unmatched_sentences.append(ill_sentence)

    # 提取主要多出来的内容，选择最长的未匹配句子
    main_extra_sentence = max(unmatched_sentences, key=len) if unmatched_sentences else ""

    # 计算编辑距离
    edit_distance = distance(main_extra_sentence, recomposed_premise)
    max_length = max(len(main_extra_sentence), len(recomposed_premise))
    similarity_ratio = 1 - (edit_distance / max_length) if max_length > 0 else 1
    similarity_threshold = 0.8  # 相似度阈值，可以根据实际情况调整
    is_similar = similarity_ratio >= similarity_threshold

    # 判断 recomposed_premise 是否几乎完全出现在 ill_query 里
    fuzzy_similarity = fuzz.partial_ratio(recomposed_premise, ill_query)
    is_premise_in_ill_query = fuzzy_similarity >= 80  # 可以根据实际情况调整阈值

    response = f"pid:{pid}\nextra: {main_extra_sentence}\nrecomposed_premise: {recomposed_premise}\n是否相同: {'是' if is_similar else '否'}\nrecomposed_premise 是否几乎完全出现在 ill_query 里: {'是' if is_premise_in_ill_query else '否'}\n------------------------------\n"
    return response


def write_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--save_frequency", type=int, default=2, help="")
    parser.add_argument("--DEBUG", type=bool, default=False, help="")
    parser.add_argument("--data_path", type=str, default="Importance_Score", help="")
    args = parser.parse_args()
    final_data_path = os.path.join('data_synthesis', args.data_path, 'process_data', 'final_data.jsonl')
    check_report_path = os.path.join('data_synthesis', args.data_path, 'process_data', 'check_report.txt')

    logging.basicConfig(
        level=logging.INFO,
        format="[check] %(message)s",
        datefmt="[%X]",
        filename="premise_critique.log"
    )
    logging.info(f"running file {__file__}")

    # 加载已经采样好的题目
    with open(final_data_path, 'r', encoding='utf-8') as f:
        problem_lines = f.readlines()
        if args.DEBUG:
            problem_lines = problem_lines[:5]  # 对前三个样本DEBUG

    def process_problem(line):
        cur_problem = json.loads(line.strip())
        return compare(cur_problem)

    # 并发处理数据
    check_report = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(tqdm(executor.map(process_problem, problem_lines), total=len(problem_lines)))
        for result in results:
            check_report.append(result)
            if len(check_report) % args.save_frequency == 0:
                write_to_file(check_report, check_report_path)

    # 处理最后一批数据
    if check_report:
        write_to_file(check_report, check_report_path)
    