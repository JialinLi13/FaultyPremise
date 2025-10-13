# contra_inf_insert
# python sample_data.py --difficulty normal --num 50 --conflict_type contra_infer_insert --source MATH-500 --random_seed 42
python sample_data.py --difficulty normal --num 5 --conflict_type contra_infer_insert --source GSM8K --random_seed 42
python sample_data.py --difficulty medium --num 5 --conflict_type contra_infer_insert --source OLYMPIAD --random_seed 42
# python sample_data.py --difficulty medium --num 50 --conflict_type contra_infer_insert --source DEEPMATH --random_seed 42
# python sample_data.py --difficulty hard --num 50 --conflict_type contra_infer_insert --source OLYMPIAD --random_seed 42
python sample_data.py --difficulty hard --num 5 --conflict_type contra_infer_insert --source DEEPMATH --random_seed 42

# contra_premise_insert
# python sample_data.py --difficulty normal --num 50 --conflict_type contra_premise_insert --source MATH-500 --random_seed 52
python sample_data.py --difficulty normal --num 5 --conflict_type contra_premise_insert --source GSM8K --random_seed 52
python sample_data.py --difficulty medium --num 5 --conflict_type contra_premise_insert --source OLYMPIAD --random_seed 52
# python sample_data.py --difficulty medium --num 50 --conflict_type contra_premise_insert --source DEEPMATH --random_seed 52
# python sample_data.py --difficulty hard --num 50 --conflict_type contra_premise_insert --source OLYMPIAD --random_seed 52
python sample_data.py --difficulty hard --num 5 --conflict_type contra_premise_insert --source DEEPMATH --random_seed 52

# irr_query_distraction
# python sample_data.py --difficulty normal --num 50 --conflict_type irr_query_distraction --source MATH-500 --random_seed 62
python sample_data.py --difficulty normal --num 5 --conflict_type irr_query_distraction --source GSM8K --random_seed 62
python sample_data.py --difficulty medium --num 5 --conflict_type irr_query_distraction --source OLYMPIAD --random_seed 62
# python sample_data.py --difficulty medium --num 50 --conflict_type irr_query_distraction --source DEEPMATH --random_seed 62
# python sample_data.py --difficulty hard --num 50 --conflict_type irr_query_distraction --source OLYMPIAD --random_seed 62
python sample_data.py --difficulty hard --num 5 --conflict_type irr_query_distraction --source DEEPMATH --random_seed 62

# flawed_solution_completion
# python sample_data.py --difficulty normal --num 50 --conflict_type flawed_solution_completion --source MATH-500 --random_seed 72
python sample_data.py --difficulty normal --num 5 --conflict_type flawed_solution_completion --source GSM8K --random_seed 72
python sample_data.py --difficulty medium --num 5 --conflict_type flawed_solution_completion --source OLYMPIAD --random_seed 72
# python sample_data.py --difficulty medium --num 50 --conflict_type flawed_solution_completion --source DEEPMATH --random_seed 72
# python sample_data.py --difficulty hard --num 50 --conflict_type flawed_solution_completion --source OLYMPIAD --random_seed 72
python sample_data.py --difficulty hard --num 5 --conflict_type flawed_solution_completion --source DEEPMATH --random_seed 72
