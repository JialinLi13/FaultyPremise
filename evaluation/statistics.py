import json
import os
from datasets import load_dataset
from collections import defaultdict
import statistics
import csv

def load_data():
    """Loads the dataset and returns a dictionary mapping pid to data entry."""
    print("Loading dataset...")
    try:
        # Use cache_dir to specify a location if needed, or let it use default
        dataset = load_dataset("Hakuno/FaultPremise", split="train")
        dataset_dict = {}
        for data in dataset:
            dataset_dict[data["pid"]] = data
        print(f"Dataset loaded with {len(dataset_dict)} entries.")
        return dataset_dict
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {} # Return empty dict on failure

def get_st_list(eval_result_lines, dataset_dict):
    """
    Processes a list of evaluation results and aggregates basic statistics.
    Handles invalid evaluation results.
    Requires dataset_dict to fetch conflict_type based on pid.
    Includes basic stats specifically for active recognition excluding flawed_solution_completion.
    """
    st_template={
        "active_result_list":[], # [1, 0, 1, ...] for True/False (all valid entries)
        "passive_result_list":[], # [1, 1, 0, ...] for True/False (all valid entries)
        "normal_correctness_list": [], # [1, 0, 1, ...] for True/False, conditional on conflict_type
        "normal_answer_len_list":[],
        "ill_answer_len_list":[],
        "ill_with_hint_answer_len_list":[],
        "difficulty_list":[], # Note: these lists are populated for *valid* entries within this group
        "conflict_type_list":[], # Note: these lists are populated for *valid* entries within this group
        "ill_eval_pid_list":[], # List of PIDs where main eval result format was unexpected

        # Basic stats for active recognition
        "active_result_list_no_fsc": [], # [1, 0, 1, ...] for True/False (excluding flawed_solution_completion)
        "active_recognition_applicable_count_no_fsc": 0, # Count of entries applicable for active_result_list_no_fsc

        "evaled_len":0 # Total number of eval results processed by this function call (input length)
    }

    for eval_result in eval_result_lines:
        st_template["evaled_len"]+=1
        pid = eval_result.get("pid", "N/A") # Get PID early

        active_eval = eval_result.get("GPT_eval_result", {}).get("active", {})
        passive_eval = eval_result.get("GPT_eval_result", {}).get("passive", {})

        active_result = active_eval.get("if_find_contradiction")
        passive_result = passive_eval.get("if_find_contradiction")

        # Get conflict_type and difficulty early for filtering and metadata lists
        current_conflict_type = "unknown"
        current_difficulty = "unknown"
        if pid != "N/A" and pid in dataset_dict:
            current_conflict_type = dataset_dict[pid].get("conflict_type", "unknown")
            current_difficulty = dataset_dict[pid].get("difficulty", "unknown")

        # Check for basic validity (True/False for active/passive)
        is_valid_core_format = (active_result in ["True","False"] and passive_result in ["True","False"])

        if not is_valid_core_format:
            st_template["ill_eval_pid_list"].append(pid)
            continue # Skip this entry for all metrics

        # If valid core format, append to main lists
        st_template["active_result_list"].append(1 if active_result=="True" else 0)
        st_template["passive_result_list"].append(1 if passive_result=="True" else 0)

        # Append to the list concluding 'Unperturbed_query'
        if current_conflict_type != "Fault_query":
            st_template["active_result_list_no_fsc"].append(1 if active_result=="True" else 0)
            st_template["active_recognition_applicable_count_no_fsc"] += 1


        # Process "normal_correctness" conditionally
        if current_conflict_type != "Fault_query":
            normal_eval = eval_result.get("GPT_eval_result", {}).get("normal", {})
            normal_correctness = normal_eval.get("correctness")
            if normal_correctness in ["True", "False"]:
                 st_template["normal_correctness_list"].append(1 if normal_correctness == "True" else 0)
            # else: # Optional: Log if correctness field is present but not True/False for a relevant type
                # print(f"Debug: PID {pid}, conflict_type {current_conflict_type}, normal_correctness value: {normal_correctness} - not adding to list.")


        # Append length stats and metadata lists for ALL valid core format entries
        normal_len = eval_result.get("normal_answer_length", {}).get("all_count", 0)
        ill_len = eval_result.get("ill_answer_length", {}).get("all_count", 0)
        ill_hint_len = eval_result.get("ill_with_hint_answer_length", {}).get("all_count", 0)

        st_template["normal_answer_len_list"].append(normal_len)
        st_template["ill_answer_len_list"].append(ill_len)
        st_template["ill_with_hint_answer_len_list"].append(ill_hint_len)

        # Append difficulty and conflict type for the *valid* entries in this specific group
        st_template["difficulty_list"].append(current_difficulty)
        st_template["conflict_type_list"].append(current_conflict_type)


    return st_template

def cal_metric(basic_st_result):
    """Calculates various metrics from the basic statistics, including length stats on success."""
    valid_count = len(basic_st_result["active_result_list"]) # Count of entries with valid core format

    # Initialize sums for efficiency calculation on success AND average lengths on success
    active_success_normal_len_sum = 0
    active_success_ill_len_sum = 0
    passive_success_normal_len_sum = 0
    passive_success_ill_with_hint_len_sum = 0
    active_success_count = 0 # Renamed for clarity (was active_success_count_for_efficiency)
    passive_success_count = 0 # Renamed for clarity (was passive_success_count_for_efficiency)


    # Ensure lists are of the same length before zipping (they should be for valid entries)
    # This check is implicitly handled by iterating up to min_len, but explicitly stating it in sums is safer
    min_len = valid_count # Since we only append to these lists for valid entries

    # Calculate sums for efficiency and average lengths on success *only* using valid indices
    for i in range(min_len):
        if basic_st_result["active_result_list"][i] == 1:
            active_success_count += 1
            active_success_normal_len_sum += basic_st_result["normal_answer_len_list"][i]
            active_success_ill_len_sum += basic_st_result["ill_answer_len_list"][i]

        if basic_st_result["passive_result_list"][i] == 1:
            passive_success_count += 1
            passive_success_normal_len_sum += basic_st_result["normal_answer_len_list"][i]
            passive_success_ill_with_hint_len_sum += basic_st_result["ill_with_hint_answer_len_list"][i]


    active_recognition_count = sum(basic_st_result["active_result_list"])
    passive_recognition_count = sum(basic_st_result["passive_result_list"])

    active_recognition_rate = active_recognition_count / valid_count if valid_count > 0 else 0.0
    passive_recognition_rate = passive_recognition_count / valid_count if valid_count > 0 else 0.0

    # --- Calculate NEW Active Recognition Rate excluding flawed_solution_completion ---
    active_result_list_no_fsc = basic_st_result.get("active_result_list_no_fsc", [])
    active_recognition_applicable_count_no_fsc = basic_st_result.get("active_recognition_applicable_count_no_fsc", 0)
    active_recognition_count_no_fsc = sum(active_result_list_no_fsc)
    active_recognition_rate_no_fsc = active_recognition_count_no_fsc / active_recognition_applicable_count_no_fsc if active_recognition_applicable_count_no_fsc > 0 else 0.0

    # --- Calculate Normal Correctness Metrics ---
    normal_correctness_list = basic_st_result.get("normal_correctness_list", [])
    normal_correctness_applicable_count = len(normal_correctness_list) # This is the count after filtering irr_query_distraction
    normal_correctness_count = sum(normal_correctness_list)
    normal_correctness_rate = normal_correctness_count / normal_correctness_applicable_count if normal_correctness_applicable_count > 0 else 0.0

    # Answer length calculations (overall based on ALL valid entries)
    normal_answer_avg_len_overall = sum(basic_st_result["normal_answer_len_list"]) / valid_count if valid_count > 0 else 0.0
    ill_answer_avg_len_overall = sum(basic_st_result["ill_answer_len_list"]) / valid_count if valid_count > 0 else 0.0
    ill_with_hint_answer_avg_len_overall = sum(basic_st_result["ill_with_hint_answer_len_list"]) / valid_count if valid_count > 0 else 0.0

    # Efficiency calculations (overall based on valid_count averages)
    active_criticism_efficiency_overall = ill_answer_avg_len_overall / normal_answer_avg_len_overall if normal_answer_avg_len_overall > 0 else 0.0
    passive_criticism_efficiency_overall = ill_with_hint_answer_avg_len_overall / normal_answer_avg_len_overall if normal_answer_avg_len_overall > 0 else 0.0

    # --- Calculate Average Lengths on Success ---
    avg_normal_len_active_success = active_success_normal_len_sum / active_success_count if active_success_count > 0 else 0.0
    avg_ill_len_active_success = active_success_ill_len_sum / active_success_count if active_success_count > 0 else 0.0
    avg_normal_len_passive_success = passive_success_normal_len_sum / passive_success_count if passive_success_count > 0 else 0.0
    avg_ill_with_hint_len_passive_success = passive_success_ill_with_hint_len_sum / passive_success_count if passive_success_count > 0 else 0.0


    # Efficiency calculations (on success based on sums for successful entries)
    active_criticism_efficiency_on_success = 0.0
    if active_success_count > 0:
        # Use the newly calculated averages on success for efficiency calculation
        active_criticism_efficiency_on_success = avg_ill_len_active_success / avg_normal_len_active_success if avg_normal_len_active_success > 0 else 0.0


    passive_criticism_efficiency_on_success = 0.0
    if passive_success_count > 0:
         # Use the newly calculated averages on success for efficiency calculation
        passive_criticism_efficiency_on_success = avg_ill_with_hint_len_passive_success / avg_normal_len_passive_success if avg_normal_len_passive_success > 0 else 0.0


    metrics = {
        "active_recognition_count": active_recognition_count,
        "passive_recognition_count": passive_recognition_count,
        "active_recognition_rate": active_recognition_rate,
        "passive_recognition_rate": passive_recognition_rate,

        "active_recognition_count_no_fsc": active_recognition_count_no_fsc,
        "active_recognition_applicable_count_no_fsc": active_recognition_applicable_count_no_fsc,
        "active_recognition_rate_no_fsc": active_recognition_rate_no_fsc,

        "normal_correctness_count": normal_correctness_count,
        "normal_correctness_rate": normal_correctness_rate,
        "normal_correctness_applicable_count": normal_correctness_applicable_count,

        # Overall Average Lengths
        "normal_answer_avg_len_overall": normal_answer_avg_len_overall,
        "ill_answer_avg_len_overall": ill_answer_avg_len_overall,
        "ill_with_hint_answer_avg_len_overall": ill_with_hint_answer_avg_len_overall,

        # Average Lengths on Success
        "normal_answer_avg_len_active_success": avg_normal_len_active_success,
        "ill_answer_avg_len_active_success": avg_ill_len_active_success,
        "normal_answer_avg_len_passive_success": avg_normal_len_passive_success,
        "ill_with_hint_answer_avg_len_passive_success": avg_ill_with_hint_len_passive_success,

        # Efficiency (Overall and on Success)
        "active_criticism_efficiency_overall": active_criticism_efficiency_overall,
        "passive_criticism_efficiency_overall": passive_criticism_efficiency_overall,
        "active_criticism_efficiency_on_success": active_criticism_efficiency_on_success,
        "passive_criticism_efficiency_on_success": passive_criticism_efficiency_on_success,

        # Counts (Success counts used for efficiency/avg len on success)
        "active_success_count": active_success_count, # Renamed
        "passive_success_count": passive_success_count, # Renamed

        "total_evaluated": basic_st_result.get("evaled_len", 0), # Total *input* count to get_st_list
        "valid_for_metrics": valid_count, # Entries with valid core True/False format
        "invalid_format_count": len(basic_st_result.get("ill_eval_pid_list", [])),
        "invalid_format_pids": basic_st_result.get("ill_eval_pid_list", []), # Keep list in dict, but exclude from CSV
    }
    return metrics

# --- Main Execution ---

eval_result_dir=os.path.join('evaluation','eval_result')
statistics_result_dir=os.path.join('evaluation','statistics')
if not os.path.exists(statistics_result_dir):
    os.makedirs(statistics_result_dir)

dataset_dict=load_data()

if not dataset_dict:
    print("Failed to load dataset. Exiting.")
    exit()

all_models_final_results = {} # To store final_results for each model

if os.path.exists(eval_result_dir) and os.path.isdir(eval_result_dir):
    for filename in os.listdir(eval_result_dir):
        if filename.endswith('.jsonl'):
            model_name=filename.split('_eval_result.jsonl')[0]
            print(f"\nStarting statistics for model: {model_name}")
            save_path=os.path.join(statistics_result_dir,f"{model_name}_eval_statistics.json")

            eval_result_lines = []
            file_path = os.path.join(eval_result_dir,filename)
            try:
                with open(file_path,'r',encoding='utf-8') as f:
                    eval_result_lines = [json.loads(line.strip()) for line in f.readlines() if line.strip()]
                print(f"Loaded {len(eval_result_lines)} evaluation results from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

            print("Calculating overall statistics...")
            overall_basic_st = get_st_list(eval_result_lines, dataset_dict)
            overall_metrics = cal_metric(overall_basic_st)
            print("Overall statistics calculated.")

            # Filter eval results *after* identifying invalid ones in overall stats
            # This ensures consistency: all subsequent groupings use only entries valid at the core level
            valid_eval_results = [
                er for er in eval_result_lines if er.get("pid") not in overall_basic_st.get("ill_eval_pid_list", [])
            ]
            print(f"Filtered down to {len(valid_eval_results)} valid evaluation entries for detailed grouping.")

            print("Grouping valid results by difficulty...")
            eval_results_by_difficulty = defaultdict(list)
            for eval_result in valid_eval_results:
                pid = eval_result.get("pid")
                # Check dataset_dict lookup explicitly for safety, though already filtered by valid_eval_results
                if pid in dataset_dict:
                    difficulty = dataset_dict[pid].get("difficulty", "unknown")
                    eval_results_by_difficulty[difficulty].append(eval_result)
            print(f"Grouped into {len(eval_results_by_difficulty)} difficulty levels.")

            print("Calculating statistics per difficulty...")
            difficulty_metrics = {}
            for difficulty, difficulty_lines in eval_results_by_difficulty.items():
                print(f"  Calculating stats for difficulty: {difficulty} ({len(difficulty_lines)} entries)")
                # Pass only the specific subset of lines for this group
                basic_st_difficulty = get_st_list(difficulty_lines, dataset_dict)
                metrics_difficulty = cal_metric(basic_st_difficulty)
                difficulty_metrics[difficulty] = metrics_difficulty
            print("Statistics per difficulty calculated.")

            print("Grouping valid results by conflict type...")
            eval_results_by_conflict_type = defaultdict(list)
            for eval_result in valid_eval_results:
                pid = eval_result.get("pid")
                 # Check dataset_dict lookup explicitly for safety
                if pid in dataset_dict:
                    conflict_type = dataset_dict[pid].get("conflict_type", "unknown")
                    eval_results_by_conflict_type[conflict_type].append(eval_result)
            print(f"Grouped into {len(eval_results_by_conflict_type)} conflict types.")

            print("Calculating statistics per conflict type...")
            conflict_type_metrics = {}
            for conflict_type, conflict_type_lines in eval_results_by_conflict_type.items():
                print(f"  Calculating stats for conflict type: {conflict_type} ({len(conflict_type_lines)} entries)")
                 # Pass only the specific subset of lines for this group
                basic_st_conflict_type = get_st_list(conflict_type_lines, dataset_dict)
                metrics_conflict_type = cal_metric(basic_st_conflict_type)
                conflict_type_metrics[conflict_type] = metrics_conflict_type
            print("Statistics per conflict type calculated.")

            # --- New Grouping: Conflict Type AND Difficulty ---
            print("Grouping valid results by conflict type and difficulty...")
            eval_results_by_conflict_type_and_difficulty = defaultdict(lambda: defaultdict(list))
            for eval_result in valid_eval_results:
                pid = eval_result.get("pid")
                if pid in dataset_dict: # Check dataset_dict lookup explicitly for safety
                    conflict_type = dataset_dict[pid].get("conflict_type", "unknown")
                    difficulty = dataset_dict[pid].get("difficulty", "unknown")
                    eval_results_by_conflict_type_and_difficulty[conflict_type][difficulty].append(eval_result)

            total_combined_entries = sum(len(lines) for difficulty_dict in eval_results_by_conflict_type_and_difficulty.values() for lines in difficulty_dict.values())
            total_combined_groups = sum(len(difficulty_dict) for difficulty_dict in eval_results_by_conflict_type_and_difficulty.values()) # Count distinct (ct, d) pairs

            print(f"Grouped into combined {total_combined_entries} entries across {len(eval_results_by_conflict_type_and_difficulty)} conflict types and {total_combined_groups} conflict type/difficulty combinations.")


            print("Calculating statistics per conflict type ...")
            conflict_type_difficulty_metrics = defaultdict(dict)
            for conflict_type, difficulty_groups in eval_results_by_conflict_type_and_difficulty.items():
                print(f"  Processing conflict type: {conflict_type}")
                for difficulty, combined_lines in difficulty_groups.items():
                     print(f"    Calculating stats for difficulty: {difficulty} ({len(combined_lines)} entries)")
                     # Pass only the specific subset of lines for this combined group
                     basic_st_combined = get_st_list(combined_lines, dataset_dict)
                     metrics_combined = cal_metric(basic_st_combined)
                     conflict_type_difficulty_metrics[conflict_type][difficulty] = metrics_combined
            print("Statistics per conflict type.")
            # --- End New Grouping ---


            print("Calculating comparative statistics across difficulties...")
            comparative_statistics_difficulty = {}
            # Define the list of metrics to include in comparative CSVs and JSON sections
            # These are typically rates, efficiencies, relevant counts, AND average lengths
            metrics_for_comparative = [
                "active_recognition_rate",
                "passive_recognition_rate",
                "active_recognition_rate_no_fsc",
                "normal_correctness_rate",

                # Add average length metrics (overall for the group, and on success for the group)
                "normal_answer_avg_len_overall", # Overall avg len for this group (e.g., easy difficulty)
                "ill_answer_avg_len_overall",
                "ill_with_hint_answer_avg_len_overall",
                "normal_answer_avg_len_active_success", # Avg len on active success for this group
                "ill_answer_avg_len_active_success",
                "normal_answer_avg_len_passive_success", # Avg len on passive success for this group
                "ill_with_hint_answer_avg_len_passive_success",

                "active_criticism_efficiency_overall",
                "passive_criticism_efficiency_overall",
                "active_criticism_efficiency_on_success",
                "passive_criticism_efficiency_on_success",
                "valid_for_metrics", # valid_for_metrics per group is useful here (total count for the group)
                "active_success_count", # Success counts per group
                "passive_success_count",
            ]

            preferred_difficulty_order = ['easy', 'medium', 'hard']
            actual_difficulties = list(difficulty_metrics.keys())
            sorted_difficulties = sorted(actual_difficulties,
                                         key=lambda d: (preferred_difficulty_order.index(d) if d in preferred_difficulty_order else len(preferred_difficulty_order), d))

            for metric_name in metrics_for_comparative:
                metric_values_across_difficulties = {}
                for difficulty in sorted_difficulties:
                    metric_value = difficulty_metrics.get(difficulty, {}).get(metric_name, None)
                    metric_values_across_difficulties[difficulty] = metric_value
                comparative_statistics_difficulty[metric_name] = metric_values_across_difficulties
            print("Comparative statistics across difficulties calculated.")

            print("Calculating comparative statistics across conflict types...")
            comparative_statistics_conflict_type = {}
            actual_conflict_types = list(conflict_type_metrics.keys())
            sorted_conflict_types = sorted(actual_conflict_types)

            for metric_name in metrics_for_comparative: # Use the same metrics_for_comparative
                # valid_for_metrics is useful here too
                metric_values_across_conflict_types = {}
                for conflict_type in sorted_conflict_types:
                    metric_value = conflict_type_metrics.get(conflict_type, {}).get(metric_name, None)
                    metric_values_across_conflict_types[conflict_type] = metric_value
                comparative_statistics_conflict_type[metric_name] = metric_values_across_conflict_types
            print("Comparative statistics across conflict types calculated.")

            # --- Define metrics for combined comparative CSVs ---
            # These are the metrics we want comparative CSVs for at the combined level (primarily efficiencies and lengths on success)
            metrics_for_combined_comparative_csv = [
                "active_criticism_efficiency_overall", # Overall efficiency for the specific (ct, d) group
                "passive_criticism_efficiency_overall",
                "active_criticism_efficiency_on_success", # Efficiency on success for the specific (ct, d) group
                "passive_criticism_efficiency_on_success",

                # Add average lengths on success for combined groups
                "normal_answer_avg_len_active_success",
                "ill_answer_avg_len_active_success",
                "normal_answer_avg_len_passive_success",
                "ill_with_hint_answer_avg_len_passive_success",

                # Also include valid_for_metrics and success counts for context in combined table
                "valid_for_metrics",
                "active_success_count",
                "passive_success_count",
            ]


            final_results = {
                "model_name": model_name,
                "overall_statistics": overall_metrics,
                "statistics_by_difficulty": difficulty_metrics,
                "comparative_statistics_by_difficulty": comparative_statistics_difficulty, # Contains key metrics per difficulty
                "statistics_by_conflict_type": conflict_type_metrics,
                "comparative_statistics_by_conflict_type": comparative_statistics_conflict_type, # Contains key metrics per conflict type
                "statistics_by_conflict_type_and_difficulty": conflict_type_difficulty_metrics, # Contains ALL metrics per combined group

               "notes": ("Metrics calculated for successfully parsed evaluation results only. "
                         "'invalid_format_count' indicates results skipped due to unexpected main structure (e.g., active/passive True/False missing). "
                         "Overall average lengths ('_overall') and efficiency ratios ('_overall') are calculated over ALL valid samples in that group (e.g., overall, per difficulty, per conflict type, per conflict type+difficulty). "
                         "Average lengths on success ('_active_success', '_passive_success') and efficiency 'on_success' are calculated ONLY for samples within that group where the model successfully identified the contradiction in that mode. "
                         "'normal_correctness_rate' is the rate of 'True' for 'GPT_eval_result.normal.correctness', EXCLUDING entries with 'conflict_type' of 'irr_query_distraction'. "
                         "The denominator for 'normal_correctness_rate' is 'normal_correctness_applicable_count'. "
                         "'active_recognition_rate_no_fsc' is the rate of 'True' for 'GPT_eval_result.active.if_find_contradiction', EXCLUDING entries with 'conflict_type' of 'flawed_solution_completion'. "
                         "The denominator for 'active_recognition_rate_no_fsc' is 'active_recognition_applicable_count_no_fsc'. "
                         "'comparative_statistics_by_difficulty' and 'comparative_statistics_by_conflict_type' sections show how key metrics vary across difficulty levels and conflict types respectively. "
                         "'statistics_by_conflict_type_and_difficulty' contains all metrics for each combination of conflict type and difficulty (only for valid entries)."
                         )
            }
            all_models_final_results[model_name] = final_results # Store results for later CSV generation

            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    # Use default=str to handle potential non-serializable items like tuples if they were stored accidentally
                    json.dump(final_results, f, ensure_ascii=False, indent=4)
                print(f"Statistics saved successfully to {save_path}")
            except Exception as e:
                print(f"Error saving statistics to {save_path}: {e}")
else:
    print(f"Evaluation result directory not found: {eval_result_dir}")

# --- CSV Generation ---
if all_models_final_results:
    print("\nGenerating comparison CSV files...")

    # 1. Overall Summary CSV
    # Get all keys from the metrics dictionary returned by cal_metric, excluding the list of PIDs and potentially applicable counts if desired
    all_possible_overall_metrics = list(cal_metric(get_st_list([], {})).keys())
    metrics_to_exclude_from_csv = ["invalid_format_pids"] # Keep applicable counts in overall
    for metric_to_exclude in metrics_to_exclude_from_csv:
        if metric_to_exclude in all_possible_overall_metrics:
             all_possible_overall_metrics.remove(metric_to_exclude)

    # Define the desired order for the overall summary CSV columns - Include new length metrics
    overall_summary_metrics_ordered = [
        # Rates (Recognition & Correctness)
        "active_recognition_rate",
        "passive_recognition_rate",
        "active_recognition_rate_no_fsc", # Filtered rate
        "normal_correctness_rate",

        # Efficiency Scores
        "active_criticism_efficiency_overall",
        "passive_criticism_efficiency_overall",
        "active_criticism_efficiency_on_success",
        "passive_criticism_efficiency_on_success",

        # Average Lengths (Overall)
        "normal_answer_avg_len_overall",
        "ill_answer_avg_len_overall",
        "ill_with_hint_answer_avg_len_overall",

        # Average Lengths (On Success) - Added these
        "normal_answer_avg_len_active_success",
        "ill_answer_avg_len_active_success",
        "normal_answer_avg_len_passive_success",
        "ill_with_hint_answer_avg_len_passive_success",

        # Counts (Success & Applicable)
        "active_recognition_count",
        "passive_recognition_count",
        "active_recognition_count_no_fsc",
        "active_recognition_applicable_count_no_fsc", # Keep applicable counts here
        "normal_correctness_count",
        "normal_correctness_applicable_count", # Keep applicable counts here
        "active_success_count", # Renamed success counts
        "passive_success_count",


        # Totals & Invalid Counts
        "valid_for_metrics",
        "invalid_format_count",
        "total_evaluated",
    ]

    # Optional safeguard: Add any metrics not explicitly listed to the end
    for metric in all_possible_overall_metrics:
        if metric not in overall_summary_metrics_ordered:
             overall_summary_metrics_ordered.append(metric)


    overall_csv_path = os.path.join(statistics_result_dir, "all_models_overall_summary.csv")
    with open(overall_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Use the ordered list for the header
        header = ["Model"] + overall_summary_metrics_ordered
        writer.writerow(header)
        for model_name, data in all_models_final_results.items():
            row = [model_name]
            # Iterate through the ordered list to fetch data for the row
            for metric in overall_summary_metrics_ordered:
                value = data["overall_statistics"].get(metric, 'N/A')
                # Format float values to 4 decimal places for readability
                if isinstance(value, float):
                    row.append(f"{value:.4f}")
                else:
                    row.append(value if value is not None else 'N/A') # Use N/A for None/missing
            writer.writerow(row)
    print(f"Overall summary CSV saved to {overall_csv_path}")

    # 2. Comparative Statistics by Difficulty CSVs (using the metrics_for_comparative list)
    # Determine all unique difficulty levels across all models, sorted
    all_difficulty_levels = set()
    preferred_difficulty_order = ['easy', 'medium', 'hard'] # from previous code
    for data in all_models_final_results.values():
        all_difficulty_levels.update(data.get("statistics_by_difficulty", {}).keys()) # Use .get for safety

    sorted_unique_difficulties = sorted(
        list(all_difficulty_levels),
        key=lambda d: (preferred_difficulty_order.index(d) if d in preferred_difficulty_order else len(preferred_difficulty_order), d)
    )

    # metrics_for_comparative defined earlier - contains rates, efficiencies, valid_for_metrics, and average lengths
    for metric_name in metrics_for_comparative:
        # Skip metrics that don't make sense in a "by difficulty" or "by conflict type" breakdown
        # 'valid_for_metrics' *is* useful per group, so keep it.
        # Overall total counts like 'total_evaluated', 'invalid_format_count' don't belong here.
        # Also skip overall average lengths as we now have overall per group
        if metric_name in ["total_evaluated", "invalid_format_count", "invalid_format_pids",
                           "normal_answer_avg_len_overall", "ill_answer_avg_len_overall", "ill_with_hint_answer_avg_len_overall"]:
            continue

        csv_path = os.path.join(statistics_result_dir, f"all_models_{metric_name}_by_difficulty.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = ["Model"] + sorted_unique_difficulties
            writer.writerow(header)
            for model_name, data in all_models_final_results.items():
                row = [model_name]
                # Use .get for safety if a model somehow missed a difficulty
                metric_data_for_model = data.get("statistics_by_difficulty", {}) # Get the stats_by_difficulty dict
                for difficulty_level in sorted_unique_difficulties:
                    # Get the specific metric value for this difficulty
                    value = metric_data_for_model.get(difficulty_level, {}).get(metric_name)
                    # Format float to a reasonable number of decimal places, e.g., 4
                    if isinstance(value, float):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(value if value is not None else 'N/A') # Use N/A for None/missing
                writer.writerow(row)
        print(f"Comparative CSV for '{metric_name}' by difficulty saved to {csv_path}")


    # 3. Comparative Statistics by Conflict Type CSVs (using the metrics_for_comparative list)
    # Determine all unique conflict types across all models, sorted
    all_conflict_types = set()
    for data in all_models_final_results.values():
        all_conflict_types.update(data.get("statistics_by_conflict_type", {}).keys()) # Use .get for safety
    sorted_unique_conflict_types = sorted(list(all_conflict_types))

    for metric_name in metrics_for_comparative: # Use the same list of metrics
        # Skip metrics that don't make sense in a "by difficulty" or "by conflict type" breakdown
        # 'valid_for_metrics' *is* useful per group, so keep it.
        # Overall total counts like 'total_evaluated', 'invalid_format_count' don't belong here.
         # Also skip overall average lengths as we now have overall per group
        if metric_name in ["total_evaluated", "invalid_format_count", "invalid_format_pids",
                           "normal_answer_avg_len_overall", "ill_answer_avg_len_overall", "ill_with_hint_answer_avg_len_overall"]:
             continue

        csv_path = os.path.join(statistics_result_dir, f"all_models_{metric_name}_by_conflict_type.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = ["Model"] + sorted_unique_conflict_types
            writer.writerow(header)
            for model_name, data in all_models_final_results.items():
                row = [model_name]
                # Use .get for safety if a model somehow missed a conflict type
                metric_data_for_model = data.get("statistics_by_conflict_type", {}) # Get the stats_by_conflict_type dict
                for conflict_type in sorted_unique_conflict_types:
                    # Get the specific metric value for this conflict type
                    value = metric_data_for_model.get(conflict_type, {}).get(metric_name)
                    if isinstance(value, float):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(value if value is not None else 'N/A') # Use N/A for None/missing
                writer.writerow(row)
        print(f"Comparative CSV for '{metric_name}' by conflict type saved to {csv_path}")

    # --- New CSVs: Comparative Statistics by Conflict Type AND Difficulty ---
    print("\nGenerating comparative CSV files by conflict type and difficulty...")

    # Determine all unique (conflict_type, difficulty) pairs across *all* models, sorted
    all_combined_keys_across_models = set()
    for data in all_models_final_results.values():
        # Collect all (ct, d) tuples from the 'statistics_by_conflict_type_and_difficulty' structure
        for ct, inner_dict in data.get("statistics_by_conflict_type_and_difficulty", {}).items():
            for d in inner_dict.keys():
                 all_combined_keys_across_models.add((ct, d))

    preferred_difficulty_order = ['easy', 'medium', 'hard'] # Ensure consistency
    sorted_unique_combined_keys = sorted(
         list(all_combined_keys_across_models),
         # Sort first by conflict_type, then by difficulty (using preferred order)
         key=lambda item: (item[0], (preferred_difficulty_order.index(item[1]) if item[1] in preferred_difficulty_order else len(preferred_difficulty_order)), item[1])
    )

    # Define the metrics we want comparative CSVs for at the combined level (primarily efficiencies and lengths on success)
    # This list was prepared earlier: metrics_for_combined_comparative_csv. It includes average lengths on success.
    # metrics_for_combined_comparative_csv = [
    #     "active_criticism_efficiency_overall",
    #     "passive_criticism_efficiency_overall",
    #     "active_criticism_efficiency_on_success",
    #     "passive_criticism_efficiency_on_success",
    #     "normal_answer_avg_len_active_success", # Added these
    #     "ill_answer_avg_len_active_success",
    #     "normal_answer_avg_len_passive_success",
    #     "ill_with_hint_answer_avg_len_passive_success",
    #     "valid_for_metrics", # Also include valid_for_metrics for context
    #     "active_success_count", # Added success counts
    #     "passive_success_count",
    # ]

    for metric_name in metrics_for_combined_comparative_csv:
        # Skip overall average lengths in combined CSVs as they are less informative than lengths on success here
        if metric_name in ["normal_answer_avg_len_overall", "ill_answer_avg_len_overall", "ill_with_hint_answer_avg_len_overall"]:
            continue

        csv_path = os.path.join(statistics_result_dir, f"all_models_{metric_name}_by_conflict_type_and_difficulty.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Header combines conflict type and difficulty, e.g., "flawed_premise - easy"
            header = ["Model"] + [f"{ct} - {d}" for ct, d in sorted_unique_combined_keys]
            writer.writerow(header)

            for model_name, data in all_models_final_results.items():
                row = [model_name]
                # Get the nested dictionary for combined stats for this model
                combined_metrics_for_model = data.get("statistics_by_conflict_type_and_difficulty", {})

                for ct, d in sorted_unique_combined_keys:
                    # Navigate the nested dictionary: combined_metrics_for_model[ct][d][metric_name]
                    # Use .get with default {} and None to handle missing keys gracefully
                    metric_value = combined_metrics_for_model.get(ct, {}).get(d, {}).get(metric_name, None)

                    if isinstance(metric_value, float):
                        row.append(f"{metric_value:.4f}")
                    else:
                        row.append(metric_value if metric_value is not None else 'N/A') # Use N/A for None/missing data point

                writer.writerow(row)
        print(f"Comparative CSV for '{metric_name}' by conflict type and difficulty saved to {csv_path}")
    # --- End New CSVs ---


else:
    print("No model data processed, skipping CSV generation.")


print("\nStatistics process finished.")