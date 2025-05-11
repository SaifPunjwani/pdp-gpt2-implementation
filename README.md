# Parameter-free Differentiable Pruning (PDP) for GPT-2

This repository implements the Parameter-free Differentiable Pruning (PDP) technique described in the paper "PDP: Parameter-free Differentiable Pruning is All You Need" by Minsik Cho, Saurabh Adya, and Devang Naik, applied to the GPT-2 language model.

## Overview

PDP is a pruning technique that:
- Is parameter-free (no additional trainable parameters)
- Is fully differentiable (allowing end-to-end training)
- Dynamically applies soft masks during training
- Gradually increases sparsity through a training schedule

## Installation

```bash
# Clone the repository
git clone https://github.com/SaifPunjwani/pdp-gpt2.git
cd pdp-gpt2

# Install dependencies
pip install torch transformers datasets wandb tqdm
```

## Usage

To train GPT-2 with PDP pruning:

```bash
python main.py --sparsity 0.9 --tau 0.02 --num_epochs 3 --use_wandb
```

### Key Arguments

- `--sparsity`: Target sparsity level (0.0-1.0)
- `--tau`: Temperature for soft masking function
- `--num_epochs`: Number of training epochs
- `--warmup_epochs`: Epochs to reach target sparsity
- `--use_wandb`: Enable logging with Weights & Biases

### Extensions

The implementation includes three possible extensions:

1. **Quantization Masking**:
   ```bash
   python main.py --sparsity 0.9 --tau 0.02 --quantize --bits 8
   ```

2. **Improved Soft Masking**:
   ```bash
   python main.py --sparsity 0.9 --tau 0.02 --improved_masking --beta 5.0
   ```

3. **Custom Optimizer**:
   ```bash
   python main.py --sparsity 0.9 --tau 0.02 --custom_optimizer
   ```

### Hyperparameter Sweeps with W&B

To run hyperparameter sweeps with Weights & Biases:

1. Create a sweep configuration (sample in `sweep_config.yaml`)
2. Initialize the sweep:
   ```bash
   wandb sweep sweep_config.yaml
   ```
3. Run the sweep agent:
   ```bash
   wandb agent SWEEP_ID
   ```

## Experiment Results

The results of the hyperparameter sweeps and extension experiments are available in the included report and presentation.

## Paper Summary

PDP introduces a parameter-free, differentiable pruning approach that:
1. Uses a soft masking function based on the magnitude of weights
2. Dynamically computes thresholds to achieve target sparsity
3. Gradually increases sparsity during training to maintain performance
4. Achieves significant compression with minimal impact on accuracy

For more details, refer to the original paper and our included report.
