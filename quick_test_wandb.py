#!/usr/bin/env python3
"""
Simple script to test W&B integration with PDP
"""
import wandb
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math

# Define a simple masking function
def pdp_soft_mask(weight, threshold, tau):
    return torch.sigmoid((weight.abs() - threshold) / tau)

def quick_test():
    """Run a minimal W&B test with PDP"""
    # Initialize W&B
    wandb.init(project="pdp-gpt2-quick-test")
    
    # Log configurations
    wandb.config.update({
        "tau": 0.02,
        "threshold": 0.5,
        "test_type": "simple_masking"
    })
    
    # Generate a random weight tensor
    weights = torch.randn(100, 100)
    threshold = 0.5
    tau = 0.02
    
    # Apply masking
    mask = pdp_soft_mask(weights, threshold, tau)
    
    # Log some metrics
    wandb.log({
        "avg_mask_value": mask.mean().item(),
        "max_mask_value": mask.max().item(),
        "min_mask_value": mask.min().item(),
        "sparsity": (mask < 0.5).float().mean().item()
    })
    
    # Log histogram of mask values
    wandb.log({"mask_histogram": wandb.Histogram(mask.flatten().numpy())})
    
    # Create a simple 2D visualization 
    # (first 10x10 slice of the weight and mask matrices)
    weight_sample = weights[:10, :10].numpy()
    mask_sample = mask[:10, :10].numpy()
    
    # Log matrices as tables
    weight_table = wandb.Table(
        columns=[f"col_{i}" for i in range(10)],
        data=[[weight_sample[i][j] for j in range(10)] for i in range(10)]
    )
    
    mask_table = wandb.Table(
        columns=[f"col_{i}" for i in range(10)],
        data=[[mask_sample[i][j] for j in range(10)] for i in range(10)]
    )
    
    wandb.log({
        "weight_matrix": weight_table,
        "mask_matrix": mask_table
    })
    
    # Log as images instead of heatmaps
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create weight heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(weight_sample, cmap='viridis')
    plt.colorbar(im)
    plt.title('Weight Values')
    plt.tight_layout()
    wandb.log({"weight_heatmap": wandb.Image(fig)})
    plt.close()
    
    # Create mask heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(mask_sample, cmap='viridis')
    plt.colorbar(im)
    plt.title('Mask Values')
    plt.tight_layout()
    wandb.log({"mask_heatmap": wandb.Image(fig)})
    
    print("Metrics logged to W&B successfully!")
    
    # Close the W&B run
    wandb.finish()

if __name__ == "__main__":
    quick_test()