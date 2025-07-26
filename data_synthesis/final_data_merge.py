import json
import os
from huggingface_hub import HfApi
from huggingface_hub import login

num_of_samples = 1200
# 将方法生成的数据合并
final_total_data_path = os.path.join( "total_final_data", "final_data.jsonl")
final_total_data = []
pid = 0
synthesis_types = ["Importance_Score",  "Unperturbed_query",  "Random_Erasing"]
for synthesis_type in synthesis_types:
    cur_final_data_path = os.path.join(
         synthesis_type, "process_data", "final_data.jsonl")
    easy_data = []
    medium_data = []
    hard_data = []
    try:
        with open(cur_final_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    cur_data = json.loads(line.strip())
                    if cur_data["difficulty"] == "normal":
                        if len(easy_data) >= num_of_samples:
                            continue
                        cur_data["pid"] = str(pid)
                        pid += 1
                        easy_data.append(cur_data)
                    elif cur_data["difficulty"] == "medium":
                        if len(medium_data) >= num_of_samples:
                            continue
                        cur_data["pid"] = str(pid)
                        pid += 1
                        medium_data.append(cur_data)
                    elif cur_data["difficulty"] == "hard":
                        if len(hard_data) >= num_of_samples:
                            continue
                        cur_data["pid"] = str(pid)
                        pid += 1
                        hard_data.append(cur_data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {cur_final_data_path}: {e}")
    except FileNotFoundError:
        print(f"File {cur_final_data_path} not found.")
    except Exception as e:
        print(f"An error occurred while reading {cur_final_data_path}: {e}")
    final_total_data.extend(easy_data)
    final_total_data.extend(medium_data)
    final_total_data.extend(hard_data)
    print(f"Total {synthesis_type} data: {len(easy_data) + len(medium_data) + len(hard_data)}")
    print(f"easy_data: {len(easy_data)}")
    print(f"medium_data: {len(medium_data)}")
    print(f"hard_data: {len(hard_data)}")
    print("---------------------------------------------------")

directory = os.path.dirname(final_total_data_path)
if not os.path.exists(directory):
    os.makedirs(directory)

# 检查并转换 relevant_premises 中的数据类型
for item in final_total_data:
    inferences = item.get('meta_info', {}).get('inferences', [])
    for inference in inferences:
        relevant_premises = inference.get('relevant premises', [])
        for i, premise in enumerate(relevant_premises):
            if isinstance(premise, str):
                try:
                    relevant_premises[i] = int(premise)
                except ValueError:
                    try:
                        relevant_premises[i] = float(premise)
                    except ValueError:
                        print(
                            f"无法将 'relevant_premises' 中的字符串 '{premise}' 转换为数值类型。")

try:
    with open(final_total_data_path, 'w', encoding='utf-8') as f:
        for item in final_total_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
except Exception as e:
    print(f"An error occurred while writing to {final_total_data_path}: {e}")

# upload to huggingface hub -------------------------------------
login(token="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")
# 初始化 API
api = HfApi()
# 上传文件
api.upload_file(
    path_or_fileobj=final_total_data_path,  # 本地文件路径
    path_in_repo="dataset.jsonl",            # 仓库中的文件路径
    repo_id="Hakuno/FaultPremise",    # 仓库 ID
    repo_type="dataset"                        # 如果是数据集，指定为 "dataset"
)