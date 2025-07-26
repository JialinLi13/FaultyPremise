export HF_ENDPOINT=https://hf-mirror.com
# 在data_synthesis目录运行bash -x synthesis_pipeline.sh
# flawed_solution_completion -------------------------------------

# 2. synthesis final data
cd Importance_Score
python faultpremise_synthesis.py --mode synthesis
# -------------------------------------------------------

# irr_query_distraction -------------------------------------


cd ../
# 2. synthesis final data
cd Unperturbed_query
python faultpremise_synthesis.py --mode synthesis
# -------------------------------------------------------

# final_data_merge -------------------------------------
cd ../
python final_data_merge.py

# -------------------------------------------------------