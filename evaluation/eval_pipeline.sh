export HF_ENDPOINT=https://hf-mirror.com
#  bash -x eval_pipeline.sh


# gpt-4o -----------------------------------------------
# infer-inference
python inference.py --model_name qwen3-8b --mode inference --save_frequency 2 --dataset_load_proc 10 --infer_proc 5
# infer-check 

python eval.py --model_name gpt-4 --mode inference --evaluator gpt-4.1 --save_frequency 2 --infer_proc 10
# eval-check 
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


