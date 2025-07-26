import copy
import os
import random
from datasets import load_dataset,concatenate_datasets
import json
import argparse


# 三级难度：normal,medium,hard
num_proc=10

data_template = {
    "pid": "",
    "ill_query": "",
    "normal_query":"",
    "conflict_type":"",
    "difficulty":"",
    "conflict":{
        "original_premise":"",
        "recomposed_premise":"",
        "conflict_reason":""
    },
    "meta_info":{
        "original_question":"",
        "reference_solution":"",
        "final_answer":"",
        "source":"",
        "source_pid":""
    }
}

def append_data_to_file(data,save_path):
    with open(save_path, "a",encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def sample_GSM8K(difficulty,conflict_type,num,start_pid,save_path,num_proc=10):
    dataset = load_dataset("openai/gsm8k", "main", num_proc=num_proc)
    dataset = dataset["test"]
    cur_pid=start_pid
    random_indices = random.sample(range(len(dataset)), num)
    my_data = []
    for i in random_indices:
        sample=dataset[i]
        cur_data=copy.deepcopy(data_template)
        cur_data["pid"]=cur_pid
        cur_pid+=1
        cur_data["difficulty"]=difficulty
        cur_data["conflict_type"]=conflict_type
        # meta info ------------------------------
        cur_data["meta_info"]["original_question"]=sample["question"]
        refernce_solution=sample["answer"]
        cur_data["meta_info"]["reference_solution"]=refernce_solution
        cur_data["meta_info"]["final_answer"]=sample["answer"].split("####")[-1].strip() if "####" in refernce_solution else None
        cur_data["meta_info"]["source"]="GSM8K"
        cur_data["meta_info"]["source_pid"]=str(i)
        # -----------------------------------------
        my_data.append(cur_data)
    append_data_to_file(my_data,save_path)
    return cur_pid

def sample_MATH500(difficulty,conflict_type,num,start_pid,save_path,num_proc=10):
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test", num_proc=num_proc)
    if difficulty=="normal":
        dataset = dataset.filter(lambda x: x["level"] == 4)
    elif difficulty=="medium":
        dataset = dataset.filter(lambda x: x["level"] == 5)
    cur_pid=start_pid
    random_indices = random.sample(range(len(dataset)), num)
    my_data = []
    for i in random_indices:
        sample=dataset[i]
        cur_data=copy.deepcopy(data_template)
        cur_data["pid"]=cur_pid
        cur_pid+=1
        cur_data["difficulty"]=difficulty
        cur_data["conflict_type"]=conflict_type
        # meta info ------------------------------
        cur_data["meta_info"]["original_question"]=sample["problem"]
        refernce_solution=sample["solution"]
        cur_data["meta_info"]["reference_solution"]=refernce_solution
        cur_data["meta_info"]["final_answer"]=sample["answer"]
        cur_data["meta_info"]["source"]="MATH-500"
        cur_data["meta_info"]["source_pid"]=sample["unique_id"]
        # -----------------------------------------
        my_data.append(cur_data)
    append_data_to_file(my_data,save_path)
    return cur_pid

def sample_Omni_MATH(difficulty,conflict_type,num,start_pid,save_path,num_proc=10):
    dataset = load_dataset("KbsdJames/Omni-MATH", split="test", num_proc=num_proc)
    if difficulty=="medium":
        dataset = dataset.filter(lambda x: x["difficulty"]>4 and x["difficulty"]<=6)
    elif difficulty=="hard":
        dataset = dataset.filter(lambda x: x["difficulty"]>6)
    cur_pid=start_pid
    random_indices = random.sample(range(len(dataset)), num)
    my_data = []
    for i in random_indices:
        sample=dataset[i]
        cur_data=copy.deepcopy(data_template)
        cur_data["pid"]=cur_pid
        cur_pid+=1
        cur_data["difficulty"]=difficulty
        cur_data["conflict_type"]=conflict_type
        # meta info ------------------------------
        cur_data["meta_info"]["original_question"]=sample["problem"]
        refernce_solution=sample["solution"]
        cur_data["meta_info"]["reference_solution"]=refernce_solution
        cur_data["meta_info"]["final_answer"]=sample["answer"]
        cur_data["meta_info"]["source"]="Omni-MATH"
        cur_data["meta_info"]["source_pid"]=str(i)
        # -----------------------------------------
        my_data.append(cur_data)
    append_data_to_file(my_data,save_path)
    return cur_pid

def sample_DEEPMATH(difficulty,conflict_type,num,start_pid,save_path,num_proc=10):
    dataset = load_dataset("zwhe99/DeepMath-103K", split="train", num_proc=num_proc)
    if difficulty=="medium":
        dataset = dataset.filter(lambda x: x["difficulty"]>4 and x["difficulty"]<=7)
    elif difficulty=="hard":
        dataset = dataset.filter(lambda x: x["difficulty"]>7)
    cur_pid=start_pid
    random_indices = random.sample(range(len(dataset)), num)
    my_data = []
    for i in random_indices:
        sample=dataset[i]
        cur_data=copy.deepcopy(data_template)
        cur_data["pid"]=cur_pid
        cur_pid+=1
        cur_data["difficulty"]=difficulty
        cur_data["conflict_type"]=conflict_type
        # meta info ------------------------------
        cur_data["meta_info"]["original_question"]=sample["question"]
        refernce_solution=sample["r1_solution_1"]
        cur_data["meta_info"]["reference_solution"]=refernce_solution
        cur_data["meta_info"]["final_answer"]=sample["final_answer"]
        cur_data["meta_info"]["source"]="DEEPMATH"
        cur_data["meta_info"]["source_pid"]=""
        # -----------------------------------------
        my_data.append(cur_data)
    append_data_to_file(my_data,save_path)
    pass

def sample_OLYMPIAD(difficulty,conflict_type,num,start_pid,save_path,num_proc=10):
    if difficulty=="medium":
        dataset = load_dataset("Hothan/OlympiadBench", name="OE_TO_maths_zh_CEE",split="train", num_proc=num_proc)
    elif difficulty=="hard":
        datasets_list = []
        for name in ["OE_TO_maths_zh_COMP","OE_TO_maths_en_COMP"]:
            dataset = load_dataset("Hothan/OlympiadBench", name=name,split="train", num_proc=num_proc)
        datasets_list.append(dataset)
        dataset=concatenate_datasets(datasets_list)
    dataset = dataset.filter(lambda x: x["is_multiple_answer"] == False)
    cur_pid=start_pid
    random_indices = random.sample(range(len(dataset)), num)
    my_data = []
    for i in random_indices:
        sample=dataset[i]
        cur_data=copy.deepcopy(data_template)
        cur_data["pid"]=cur_pid
        cur_pid+=1
        cur_data["difficulty"]=difficulty
        cur_data["conflict_type"]=conflict_type
        # meta info ------------------------------
        cur_data["meta_info"]["original_question"]=sample["question"]
        refernce_solution=sample["solution"][0]
        cur_data["meta_info"]["reference_solution"]=refernce_solution
        cur_data["meta_info"]["final_answer"]=sample["final_answer"][0]
        cur_data["meta_info"]["source"]="OLYMPIAD"
        cur_data["meta_info"]["source_pid"]=str(sample["id"])
        my_data.append(cur_data)
    append_data_to_file(my_data,save_path)
    return cur_pid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample datasets")
    parser.add_argument("--difficulty", type=str, default="normal", help="Difficulty level: normal, medium, hard")
    parser.add_argument("--num", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--conflict_type", type=str, default="contra_premise_insert", help="contra_premise_insert,contra_inf_insert,irr_query_distraction,flawed_solution_completion")
    parser.add_argument("--source", type=str, default="GSM8K", help="Source dataset: GSM8K, MATH-500, OLYMPIAD, Omni-MATH")
    parser.add_argument("--load_proc", type=int, default=10, help="dataset load process")
    parser.add_argument("--random_seed", type=int, default=42, help="random_seed")
    args = parser.parse_args()
    random.seed(args.random_seed)
    save_path=os.path.join(args.conflict_type,"process_data","origin_problem.jsonl") # 原题存放路径
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    # 获取start pid
    if os.path.exists(save_path):
        with open(save_path, "r",encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) > 0:
                last_line = lines[-1]
                last_data = json.loads(last_line)
                cur_pid = int(last_data["pid"]) + 1
            else:
                cur_pid = 0
    else:
        cur_pid = 0

    if args.source=="GSM8K":
        cur_pid=sample_GSM8K(args.difficulty,args.conflict_type,args.num,cur_pid,save_path)
    elif args.source=="MATH-500":
        cur_pid=sample_MATH500(args.difficulty,args.conflict_type,args.num,cur_pid,save_path)
    elif args.source=="DEEPMATH":
        cur_pid=sample_DEEPMATH(args.difficulty,args.conflict_type,args.num,cur_pid,save_path)
    elif args.source=="OLYMPIAD":
        cur_pid=sample_OLYMPIAD(args.difficulty,args.conflict_type,args.num,cur_pid,save_path)
    elif args.source=="Omni-MATH":
        cur_pid=sample_Omni_MATH(args.difficulty,args.conflict_type,args.num,cur_pid,save_path)
