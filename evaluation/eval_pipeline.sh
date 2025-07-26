export HF_ENDPOINT=https://hf-mirror.com
# 需要在主目录运行 bash -x eval_pipeline.sh

# 普通模型

# gpt-4o -----------------------------------------------(正常输出，非流式，完成推理)
# infer-inference
python inference.py --model_name qwen3-8b --mode inference --save_frequency 2 --dataset_load_proc 10 --infer_proc 5
# infer-check 直到没有样本被遗漏

python eval.py --model_name gpt-4 --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10
# eval-check 直到没有样本被遗漏
# statistics
python statistics.py
# ----------------------------------------------------------
python ./evaluation/inference.py --model_name deepseek-ai/DeepSeek-R1-0528 --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 5 --stream --temperature 0.7 --top_p 0.8

(Jialinli)python inference.py --model_name deepseek-ai/DeepSeek-R1-0528 --mode inference --save_frequency 2 --dataset_load_proc 10 --infer_proc 5
          python inference.py --model_name deepseek-ai/DeepSeek-R1 --mode inference --save_frequency 1 --dataset_load_proc 10 --infer_proc 1

(hakuno)python inference.py --model_name Qwen/Qwen3-32B --mode inference --save_frequency 2 --dataset_load_proc 10 --infer_proc 5
(hakuno)python inference.py --model_name llama-3-70b-instruct --mode inference --save_frequency 2 --dataset_load_proc 10 --infer_proc 5


python eval.py --model_name DeepSeek-R1 --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10


python eval.py --model_name llama-3-70b-instruct --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10


