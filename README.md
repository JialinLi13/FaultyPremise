# FaultyPremise

<div align="center">
  <a href="https://arxiv.org/abs/2508.03622">
    <strong>📃 Paper</strong>
  </a>
  •
  <a href="">
    <strong>🤗 Dataset</strong>
  </a>
  •
  <a href="https://github.com/JialinLi13/FaultyPremise">
    <strong>🖥️ Code</strong>
  </a>
</div>

## Updates
[2025/08] We released codes for this project.

## Contents
- [Introduction](#introduction)
- [Contribution](#contribution)
- [Data Construction](#data-construction)
- [Install](#install)
- [Run Code](#run-code)
- [Citation](#citation)

## Introduction
With the advancement of code generation capabilities in large language models (LLMs), their reliance on input premises has intensified. When users provide inputs containing faulty premises, the probability of code generation hallucinations rises significantly, exposing deficiencies in their self-scrutiny capabilities. This paper proposes Faulty Premises Bench (FPBench), the first code generation evaluation framework targeting faulty premises. By systematically constructing three categories of faulty premises and integrating multidimensional evaluation metrics, it conducts in-depth assessments of 15 representative LLMs. T



## Contribution

- We are the first to propose a comprehensive benchmark specifically designed to assess the self-scrutiny capabilities of LLMs when confronted with faulty premises in code generation tasks.
- We have developed innovative data construction methods, including those based on importance score analysis, random erasure, and the introduction of irrelevant information perturbations. These approaches enable us to systematically construct and expand a test set targeting faulty premises (comprising 1,800 problems in total) from existing code datasets.
- We have designed a unique set of evaluation dimensions, including ”proactive error identification rate”, ”passive error identification rate”, and ”self-scrutiny overhead ratio”. These metrics aim to comprehensively quantify the model’s ability to identify, process, and respond to faulty premises, as well as its resource consumption.


## Data Construction

We construct **PCBench** to systematically evaluate LLMs' premise critique abilities for erroneous inputs via a structured process:  
1. **Error Categories**: Define 4 types of premise errors to assess model capabilities in identifying flawed inputs.  
2. **Difficulty Levels**:  
   - Normal: From GSM8K dataset  
   - Medium: Adapted from Chinese College Entrance Examination (OlympiadBench)  
   - Difficult: From Omni-MATH (difficulty >6)  
3. **Problem Variants** for each base problem (error category + difficulty):  
   - **Original Problem**: Correct premises (baseline).  
   - **Flawed Problem**: Intentional errors in premises (to test autonomous critique).  
   - **Flawed Problem with Explicit Instruction**: Adds prompts to check for errors (comparative reference).  

**Scale**: 100 base problems per error-difficulty combination → 1,200 base problems → 3,600 problems (3 variants each).  
Designed to analyze how error type and task complexity impact premise critique ability.
<p align="center" width="90%">
<a ><img src="resources/pipeline.png" alt="construction" style="width: 60%; min-width: 500px; display: block; margin: auto;"></a>
</p>

## Results

<p align="center" width="80%">
<a ><img src="resources/results.png" alt="results" style="width: 60%; min-width: 550px; display: block; margin: auto;"></a>
</p>


## Run Code

### Inference
Run following commad to get LMM's responses.

```bash
python data_synthesis\inference.py --model_name <model_name>
```

### Evaluation
Run following commad to get o3's evaluation result to corresponding responses.

```bash
python evaluation\evaluate.py --model_folder <model_responses> --model_name <model_name>
```

## Citation
```
@misc{yang2025largemultimodalmodelsactively,
      title={Can Large Multimodal Models Actively Recognize Faulty Inputs? A Systematic Evaluation Framework of Their Input Scrutiny Ability}, 
      author={Haiqi Yang and Jinzhe Li and Gengxu Li and Yi Chang and Yuan Wu},
      year={2025},
      eprint={2508.04017},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.04017}, 
}
```
Please cite our paper if you find our research and code useful.


