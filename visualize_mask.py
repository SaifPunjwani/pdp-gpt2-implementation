import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import argparse

def pdp_soft_mask(weight, threshold, tau):
    """Standard PDP soft masking function."""
    return torch.sigmoid((weight.abs() - threshold) / tau)

def improved_soft_mask(weight, threshold, tau, beta=5.0):
    """Enhanced soft masking function with smoother gradient transition."""
    sig = torch.sigmoid((weight.abs() - threshold) / tau)
    tanh_comp = 0.5 * (torch.tanh(beta * (weight.abs() - threshold) / tau) + 1)
    return 0.7 * sig + 0.3 * tanh_comp

def visualize_masking_functions():
    """Visualize different masking functions and their gradients."""
    # Create weight values for visualization
    weights = torch.linspace(-1.0, 1.0, 1000)
    threshold = 0.5
    
    plt.figure(figsize=(15, 10))
    
    # Test different tau values for standard mask
    taus = [0.01, 0.02, 0.05, 0.1]
    
    # Plot 1: Standard masking function with different taus
    plt.subplot(2, 2, 1)
    for tau in taus:
        mask = pdp_soft_mask(weights, threshold, tau)
        plt.plot(weights.numpy(), mask.numpy(), label=f'τ={tau}')
    
    plt.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=-threshold, color='gray', linestyle='--', alpha=0.5)
    plt.title('Standard PDP Soft Mask')
    plt.xlabel('Weight Magnitude')
    plt.ylabel('Mask Value')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Gradient of standard masking function
    plt.subplot(2, 2, 2)
    for tau in taus:
        # Manual gradient computation
        weights_grad = weights.clone().requires_grad_(True)
        mask = pdp_soft_mask(weights_grad, threshold, tau)
        
        # Use autograd to compute gradients
        dummy_loss = mask.sum()
        dummy_loss.backward()
        
        plt.plot(weights.numpy(), weights_grad.grad.numpy(), label=f'τ={tau}')
    
    plt.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=-threshold, color='gray', linestyle='--', alpha=0.5)
    plt.title('Gradient of Standard PDP Soft Mask')
    plt.xlabel('Weight Magnitude')
    plt.ylabel('Gradient Value')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Compare standard vs improved masking
    plt.subplot(2, 2, 3)
    tau = 0.02
    beta_values = [2.0, 5.0, 10.0]
    
    # Standard mask
    std_mask = pdp_soft_mask(weights, threshold, tau)
    plt.plot(weights.numpy(), std_mask.numpy(), label='Standard', linewidth=2)
    
    # Improved masks with different beta values
    for beta in beta_values:
        imp_mask = improved_soft_mask(weights, threshold, tau, beta)
        plt.plot(weights.numpy(), imp_mask.numpy(), label=f'Improved β={beta}')
    
    plt.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=-threshold, color='gray', linestyle='--', alpha=0.5)
    plt.title('Standard vs. Improved Mask (τ=0.02)')
    plt.xlabel('Weight Magnitude')
    plt.ylabel('Mask Value')
    plt.grid(True)
    plt.legend()
    
    # Plot 4: Gradient comparison
    plt.subplot(2, 2, 4)
    
    # Standard mask gradient
    weights_grad = weights.clone().requires_grad_(True)
    std_mask = pdp_soft_mask(weights_grad, threshold, tau)
    dummy_loss = std_mask.sum()
    dummy_loss.backward()
    std_grad = weights_grad.grad.clone()
    
    plt.plot(weights.numpy(), std_grad.numpy(), label='Standard', linewidth=2)
    
    # Improved mask gradients
    for beta in beta_values:
        weights_grad = weights.clone().requires_grad_(True)
        imp_mask = improved_soft_mask(weights_grad, threshold, tau, beta)
        dummy_loss = imp_mask.sum()
        dummy_loss.backward()
        
        plt.plot(weights.numpy(), weights_grad.grad.numpy(), label=f'Improved β={beta}')
    
    plt.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=-threshold, color='gray', linestyle='--', alpha=0.5)
    plt.title('Gradient Comparison (τ=0.02)')
    plt.xlabel('Weight Magnitude')
    plt.ylabel('Gradient Value')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mask_comparison.png')
    plt.show()

def visualize_masking_functions_wandb():
    """Visualize mask functions and upload to W&B."""
    # Initialize wandb
    wandb.init(project="pdp-gpt2-visualize", name="masking_functions")
    
    # Create weight values for visualization
    weights = torch.linspace(-1.0, 1.0, 1000)
    threshold = 0.5
    
    # Test different tau values
    taus = [0.01, 0.02, 0.05, 0.1]
    
    # Generate mask values for standard function
    mask_data = []
    for tau in taus:
        mask = pdp_soft_mask(weights, threshold, tau).numpy()
        for i in range(0, len(weights), 10):  # Downsample for wandb
            mask_data.append([weights[i].item(), mask[i], f"τ={tau}"])
    
    # Create and log wandb table for standard mask
    mask_table = wandb.Table(
        columns=["weight", "mask_value", "tau"],
        data=mask_data
    )
    wandb.log({"standard_mask": wandb.plot.line(
        mask_table, "weight", "mask_value", 
        title="Standard PDP Soft Mask",
        stroke="tau"
    )})
    
    # Generate gradient data for standard function
    grad_data = []
    for tau in taus:
        weights_grad = weights.clone().requires_grad_(True)
        mask = pdp_soft_mask(weights_grad, threshold, tau)
        dummy_loss = mask.sum()
        dummy_loss.backward()
        grads = weights_grad.grad.numpy()
        
        for i in range(0, len(weights), 10):  # Downsample
            grad_data.append([weights[i].item(), grads[i], f"τ={tau}"])
    
    # Create and log wandb table for standard gradient
    grad_table = wandb.Table(
        columns=["weight", "gradient", "tau"],
        data=grad_data
    )
    wandb.log({"standard_gradient": wandb.plot.line(
        grad_table, "weight", "gradient", 
        title="Gradient of Standard PDP Soft Mask",
        stroke="tau"
    )})
    
    # Compare standard vs improved
    tau = 0.02
    beta_values = [2.0, 5.0, 10.0]
    
    # Generate comparison data
    comparison_data = []
    
    # Standard mask
    std_mask = pdp_soft_mask(weights, threshold, tau).numpy()
    for i in range(0, len(weights), 10):
        comparison_data.append([weights[i].item(), std_mask[i], "Standard"])
    
    # Improved masks
    for beta in beta_values:
        imp_mask = improved_soft_mask(weights, threshold, tau, beta).numpy()
        for i in range(0, len(weights), 10):
            comparison_data.append([weights[i].item(), imp_mask[i], f"Improved β={beta}"])
    
    # Create and log wandb table for comparison
    comparison_table = wandb.Table(
        columns=["weight", "mask_value", "mask_type"],
        data=comparison_data
    )
    wandb.log({"mask_comparison": wandb.plot.line(
        comparison_table, "weight", "mask_value", 
        title="Standard vs. Improved Mask (τ=0.02)",
        stroke="mask_type"
    )})
    
    # Generate the matplotlib figures for local viewing
    plt_fig = visualize_masking_functions()
    wandb.log({"mask_plots": wandb.Image('mask_comparison.png')})
    
    wandb.finish()
    return plt_fig

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize PDP masking functions")
    parser.add_argument("--use_wandb", action="store_true", help="Upload visualizations to W&B")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.use_wandb:
        visualize_masking_functions_wandb()
    else:
        visualize_masking_functions()