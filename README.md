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
With the advancement of code generation capabilities in large language models (LLMs), their reliance on input premises has intensified. When users provide inputs containing faulty premises, the probability of code generation hallucinations rises significantly, exposing deficiencies in their self-scrutiny capabilities. This paper proposes Faulty Premises Bench (FPBench), the first code generation evaluation framework targeting faulty premises. By systematically constructing three categories of faulty premises and integrating multidimensional evaluation metrics, it conducts in-depth assessments of 15 representative LLMs. 



## Contribution

- We are the first to propose a comprehensive benchmark specifically designed to assess the self-scrutiny capabilities of LLMs when confronted with faulty premises in code generation tasks.
- We have developed innovative data construction methods, including those based on importance score analysis, random erasure, and the introduction of irrelevant information perturbations. These approaches enable us to systematically construct and expand a test set targeting faulty premises (comprising 1,800 problems in total) from existing code datasets.
- We have designed a unique set of evaluation dimensions, including ”proactive error identification rate”, ”passive error identification rate”, and ”self-scrutiny overhead ratio”. These metrics aim to comprehensively quantify the model’s ability to identify, process, and respond to faulty premises, as well as its resource consumption.


## Data Construction

We randomly collected 600 pieces of raw data from two datasets: HumanEval, MBPP+. Based on the following introduced three different types of erroneous premises defined by us, we reconstructed them into FPbench. Each one is designed to evaluate different aspects of the model’s ability to recognize and reason about flawed input. By constructing 600 base problems for each error type, we
obtained a total of 1,800 unique base problems. This structure and scalable design enables rigorous evaluation of how self-scrutiny capabilities are influenced by error types and task complexity.

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
@article{li2025refining,
  title={Refining Critical Thinking in LLM Code Generation: A Faulty Premise-based Evaluation Framework},
  author={Li, Jialin and Li, Jinzhe and Li, Gengxu and Chang, Yi and Wu, Yuan},
  journal={arXiv preprint arXiv:2508.03622},
  year={2025}
}
```
Please cite our paper if you find our research and code useful.





