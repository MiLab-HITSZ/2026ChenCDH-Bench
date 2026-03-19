---
language:
- en
- zh
license: mit
task_categories:
- question-answering
- multiple-choice
tags:
- multimodal
- vision-language
- counterfactual-reasoning
- commonsense-reasoning
- benchmark
size_categories:
- 100<n<1K
---

# CDH-Bench: A Comprehensive Benchmark for Counterfactual vs. Commonsense Reasoning in Multimodal Models

CDH-Bench is a specialized benchmark designed to evaluate the robust reasoning capabilities of Vision-Language Models (VLMs) by comparing their performance on **Counterfactual (CF)** scenarios against **Commonsense (CS)** scenarios.

The benchmark focuses on identifying whether models rely on simple pattern matching (shortcuts) or possess a deeper understanding of visual and relational logic.

## Dataset Structure

The benchmark is provided in JSONL format with corresponding images.

### Data Files
- **`data/benchmark.jsonl`**: The core dataset containing 128 pairs of CF/CS questions and ground truth.
- **`data/images/`**: The image folder organized by subcategories and pair IDs.
- **`example_results/`**: Pre-computed evaluation outputs for testing metrics scripts.

### JSONL Entry Example
```json
{
  "pair_id": "Pair_ID",
  "category": "Primary Category",
  "subcategory": "Secondary Category",
  "direct_qa": {
    "question": "...",
    "commonsense_gt": "yes",
    "counterfactual_gt": "no"
  },
  "multiple_choice": {
    "question": "...",
    "options": ["A...", "B...", "C...", "D..."],
    "commonsense_gt": "C",
    "counterfactual_gt": "A"
  }
}
```

## Core Scripts

We provide essential scripts for data loading and evaluation metrics:

### 1. Data Loading
- **`scripts/cdh_bench_loader.py`**: A utility class to load the JSONL data and filter items by category or task format.

### 2. Evaluation & Metrics
- **`scripts/evaluate_cdh_bench.py`**: The main evaluation script. It handles model inference (via API) and calculates basic accuracy.
- **`scripts/summarize_all_results.py`**: Aggregates raw outputs and calculates key robustness metrics:
  - **CS Acc**: Accuracy on Commonsense samples.
  - **CF Acc**: Accuracy on Counterfactual samples.
  - **Gap**: $|CS\_Acc - CF\_Acc|$.
  - **CCR (Consistent Correctness Rate)**: Rate of correctly answering both CS and CF versions of a pair.

## Quick Start with Example Results

We provide pre-computed evaluation results for **Qwen3-VL-32B** models in `example_results/`. You can quickly test the metrics aggregation script without running a full evaluation:

```bash
# Run summary script on example results
python scripts/summarize_all_results.py --results_dir example_results/
```

## How to Run Evaluation

1. Prepare your model configuration (refer to `models_config_example.json`).
2. Run the evaluation:
```bash
python scripts/evaluate_cdh_bench.py \
    --data_path data/benchmark.jsonl \
    --images_root data/images/ \
    --models_config models_config_example.json \
    --task qa  # options: qa, mc
```

3. Summarize the results:
```bash
python scripts/summarize_all_results.py --results_dir results/
```

## Citation

```bibtex
@article{cdh_bench2024,
  title={CDH-Bench: Evaluating Counterfactual and Commonsense Reasoning in Vision-Language Models},
  author={...},
  journal={...},
  year={2024}
}
```
