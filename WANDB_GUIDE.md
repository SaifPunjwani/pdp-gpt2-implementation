# Weights & Biases (W&B) Integration Guide

This document explains how to use Weights & Biases (W&B) for logging and hyperparameter sweeps with the PDP implementation.

## Initial Setup

1. **Install W&B**:
   ```bash
   pip install wandb
   ```

2. **Login to W&B**:
   ```bash
   wandb login
   ```
   You'll be prompted to enter your API key from https://wandb.ai/authorize

## Running Experiments with W&B Logging

### Quick Tests with Visualization

To visualize masking functions and upload results to W&B:

```bash
python visualize_mask.py --use_wandb
```

This will create interactive plots in your W&B project showing:
- Standard PDP masking function with different tau values
- Gradients of the masking functions 
- Comparison between standard and improved masking

### Testing PDP at Different Sparsity Levels

To test PDP on GPT-2 with W&B logging:

```bash
python test_pdp.py --use_wandb --tau 0.02 --improved_masking --beta 5.0
```

This will:
- Create a W&B run with your parameters
- Test different sparsity levels from 0% to 95%
- Log metrics like perplexity, loss, and actual sparsity
- Show text generation results at each sparsity level
- Create summary visualizations

## Running Hyperparameter Sweeps

### 1. Create a Sweep

For a quick test of different masking parameters:

```bash
wandb sweep sweep_config_test.yaml
```

For the full training sweep:

```bash
wandb sweep sweep_config.yaml
```

This will output a sweep ID like: `wandb/pdp-gpt2-sweep/abcd1234`

### 2. Run the Sweep Agent

For the quick test sweep:

```bash
python sweep_agent.py --sweep_id YOUR_SWEEP_ID
```

For the full training sweep:

```bash
wandb agent YOUR_SWEEP_ID
```

### 3. Viewing Results

1. Go to your W&B project page
2. Navigate to the "Sweeps" tab
3. Click on your sweep to see:
   - Parameter importance analysis
   - Parallel coordinates plot
   - Individual run results

## Creating Custom Sweep Configurations

You can modify the provided sweep configs or create new ones:

1. `sweep_config_test.yaml`: Quick grid search of masking parameters
2. `sweep_config.yaml`: Full hyperparameter search for training

To create a custom sweep:
1. Create a new YAML file
2. Specify the search method (grid, random, bayes)
3. Define parameters and their ranges
4. Run `wandb sweep` with your new file

## Best Practices

1. **Organize Projects**:
   - Use descriptive project names
   - Tag runs with meaningful labels

2. **Log All Relevant Metrics**:
   - Perplexity and loss
   - Actual vs. target sparsity
   - Training time
   - Generated text quality

3. **Save Artifacts**:
   - Model checkpoints
   - Generated text examples
   - Visualizations

For more information, visit: https://docs.wandb.ai/