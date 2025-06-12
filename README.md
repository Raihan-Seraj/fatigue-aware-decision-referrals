# Fatigue-Aware Decision Referrals in Human-Automation Teams

This repository contains the implementation of fatigue-aware decision referral algorithms for human-automation teams. The code implements an Approximate Dynamic Programming (ADP) approach and compares it with the Kesav algorithm for optimal task allocation under operator fatigue.

## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Scripts Overview](#scripts-overview)
- [Arguments Reference](#arguments-reference)
- [Output Structure](#output-structure)



## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd fatigue-aware-decision-referrals
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv fatigue_env

# Activate virtual environment
# On macOS/Linux:
source fatigue_env/bin/activate

# On Windows:
fatigue_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Test the installation
python -c "import numpy, scipy, matplotlib, tqdm; print('All dependencies installed successfully!')"
```

## 📖 Usage

The evaluation pipeline consists of four main steps that **must be executed in order**:

### Step 1: Run Approximate Dynamic Programming 🧮

This computes the optimal value function using backward induction.

```bash
python src/approximate_dp.py
```

**⚠️ Important**: This step must be completed first as it generates the value function required by subsequent scripts.

### Step 2: Single Run Evaluation 📊

Evaluates algorithm performance with different initial fatigue states.

```bash
python src/eval_single_run.py
```

### Step 3: Robustness Analysis 🔧

Tests algorithm robustness under perturbed fatigue models.

```bash
python src/evaluate_perturbed_fatigue.py
```

### Step 4: Generate Plots 📈

Creates visualization plots from the evaluation results.

```bash
python src/plotting_scripts.py
```

## 📁 Scripts Overview

| Script | Purpose | Dependencies | Output |
|--------|---------|--------------|---------|
| `approximate_dp.py` | Computes optimal value function via ADP | None | Value function, parameters |
| `eval_single_run.py` | Single trajectory analysis | Value function | Trajectory data |
| `evaluate_perturbed_fatigue.py` | Robustness testing | Value function | Robustness metrics |
| `plotting_scripts.py` | Visualization generation | All evaluation results | Plots and figures |

## ⚙️ Arguments Reference

### approximate_dp.py

#### Algorithm Parameters
- `--num_expectation_samples` (int, default=500): Number of Monte Carlo samples for expectation approximation
- `--horizon` (int, default=20): Time horizon length for the dynamic program
- `--num_eval_runs` (int, default=1000): Number of Monte Carlo runs for performance evaluation

#### Problem Setup
- `--prior` (list, default=[0.5, 0.5]): Prior probabilities [P(H0), P(H1)] for the two hypotheses
- `--num_tasks_per_batch` (int, default=20): Number of tasks in each decision batch
- `--num_bins_fatigue` (int, default=10): Number of discretization bins for fatigue states
- `--sigma_a` (float, default=0.5): Standard deviation of automation observation noise

#### Hypothesis Values
- `--H0` (int, default=0): Value representing the null hypothesis state
- `--H1` (int, default=3): Value representing the alternative hypothesis state

#### Cost Parameters
- `--ctp` (float, default=0.0): Cost of true positive classification
- `--ctn` (float, default=0.0): Cost of true negative classification  
- `--cfp` (float, default=1.0): Cost of false positive classification
- `--cfn` (float, default=1.0): Cost of false negative classification
- `--cr` (float, default=0.0): Cost of task referral to human operator

#### Fatigue Model Parameters
- `--alpha_tp` (float, default=0.087): Fatigue impact coefficient for true positive probability
- `--beta_tp` (float, default=0.043): Workload impact coefficient for true positive probability
- `--alpha_fp` (float, default=0.1): Fatigue impact coefficient for false positive probability
- `--beta_fp` (float, default=0.033): Workload impact coefficient for false positive probability
- `--gamma_tp` (float, default=0.0): Additional parameter for true positive probability
- `--gamma_fp` (float, default=0.0): Additional parameter for false positive probability
- `--fatigue_model` (str, default='fatigue_model_1'): Fatigue model type selection

#### Execution Parameters
- `--results_path` (str, default='results/'): Base directory for saving results
- `--run_eval_only` (bool, default=False): Skip ADP computation and run evaluation only

### Example Usage with Custom Parameters

```bash
# Run with custom parameters
python src/approximate_dp.py \
    --horizon 30 \
    --num_expectation_samples 1000 \
    --num_tasks_per_batch 15 \
    --alpha_tp 0.1 \
    --beta_tp 0.05 \
    --cfp 2.0 \
    --cfn 2.0
```

## 📂 Output Structure

The pipeline generates the following directory structure:

```
results/
        ├── params.json                         # Configuration parameters
        ├── V_func.pkl                          # Value function
        ├── cost_comparison/                    # Performance metrics
        │   ├── all_cum_cost_k.pkl
        │   ├── all_cum_cost_adp.pkl
        │   └── *.npy files
        └── plot_analysis/                      # Trajectory analysis
            ├── all_fatigue_k.pkl
            ├── all_fatigue_adp.pkl
            └── *.pdf plots
├── results_single_run/                         # Single run trajectories
│   └── run_*/
│       ├── fatigue_high/
│       └── fatigue_low/
└── results_perturbed_fatigue/                  # Robustness analysis
    ├── params.json
    └── model_*/
        └── cost_comparison/
```


## 🏃‍♂️ Quick Start Example

```bash
# 1. Activate environment
source fatigue_env/bin/activate

# 2. Run complete pipeline with default parameters
python src/approximate_dp.py
python src/eval_single_run.py  
python src/evaluate_perturbed_fatigue.py
python src/plotting_scripts.py

# 3. Check results
ls results/
```

## 📊 Expected Runtime

Approximate execution times on a modern laptop:

| Script | Default Parameters | Reduced Parameters |
|--------|-------------------|-------------------|
| `approximate_dp.py` | 10-30 minutes | 2-5 minutes |
| `eval_single_run.py` | 5-10 minutes | 1-2 minutes |
| `evaluate_perturbed_fatigue.py` | 15-45 minutes | 3-8 minutes |
| `plotting_scripts.py` | 1-2 minutes | 30 seconds |




