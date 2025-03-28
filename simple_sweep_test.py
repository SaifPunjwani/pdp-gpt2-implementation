#!/usr/bin/env python3
"""
Simple W&B sweep test for PDP
"""
import torch
import math
import wandb
import numpy as np
import matplotlib.pyplot as plt

def pdp_soft_mask(weight, threshold, tau):
    """Standard PDP soft mask function."""
    return torch.sigmoid((weight.abs() - threshold) / tau)

def improved_soft_mask(weight, threshold, tau, beta=5.0):
    """Improved PDP mask function."""
    sig = torch.sigmoid((weight.abs() - threshold) / tau)
    tanh_comp = 0.5 * (torch.tanh(beta * (weight.abs() - threshold) / tau) + 1)
    return 0.7 * sig + 0.3 * tanh_comp

def run_test():
    """Run a simple test for W&B sweep."""
    wandb.init()
    
    # Get config parameters
    tau = wandb.config.tau
    threshold = wandb.config.threshold
    improved = wandb.config.improved_masking
    beta = wandb.config.beta if hasattr(wandb.config, 'beta') else 5.0
    
    # Create weight values
    weights = torch.linspace(-1.0, 1.0, 1000)
    
    # Apply masking function
    if improved:
        mask = improved_soft_mask(weights, threshold, tau, beta).numpy()
        mask_type = f"improved_beta{beta}"
    else:
        mask = pdp_soft_mask(weights, threshold, tau).numpy()
        mask_type = "standard"
    
    # Calculate metrics
    sparsity = (mask < 0.5).mean()
    avg_mask = mask.mean()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(weights.numpy(), mask)
    ax.axvline(x=threshold, color='red', linestyle='--')
    ax.axvline(x=-threshold, color='red', linestyle='--')
    ax.set_title(f"Mask Function (tau={tau}, threshold={threshold}, {mask_type})")
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Mask Value")
    ax.grid(True)
    
    # Log everything to wandb
    wandb.log({
        "sparsity": sparsity,
        "avg_mask_value": avg_mask,
        "mask_visualization": wandb.Image(fig),
        "mask_histogram": wandb.Histogram(mask)
    })
    
    plt.close(fig)
    
    return float(avg_mask)

if __name__ == "__main__":
    run_test()