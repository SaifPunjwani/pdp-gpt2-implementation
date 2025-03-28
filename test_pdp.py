import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math
import argparse
from tqdm import tqdm
import wandb

# ==============================
# Parameter-free Differentiable Pruning (PDP) Implementation
# ==============================

def pdp_soft_mask(weight, threshold, tau):
    """
    Compute the soft mask using a sigmoid-like function centered at threshold.
    
    Args:
        weight (torch.Tensor): The weight tensor
        threshold (float): Pruning threshold (per-layer, per-step)
        tau (float): Temperature controlling sharpness of transition
        
    Returns:
        mask (torch.Tensor): Values in [0, 1] representing soft mask
    """
    return torch.sigmoid((weight.abs() - threshold) / tau)

def compute_threshold(weight, target_sparsity):
    """
    Compute threshold that achieves desired sparsity for given weight tensor.
    
    Args:
        weight (torch.Tensor): Weight tensor to compute threshold for
        target_sparsity (float): Target sparsity level in [0, 1]
        
    Returns:
        float: Threshold value that achieves target sparsity
    """
    num_elements = weight.numel()
    num_prune = int(num_elements * target_sparsity)
    if num_prune == 0:
        return 0.0
    flat = weight.abs().flatten()
    threshold = torch.kthvalue(flat, num_prune).values.item()
    return threshold

def get_sparsity(model):
    """
    Calculate the current sparsity of the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        float: Current sparsity level of the model
    """
    total_zeros = 0
    total_elements = 0
    
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            total_zeros += torch.sum(param.data.abs() < 1e-6).item()
            total_elements += param.numel()
            
    if total_elements == 0:
        return 0.0
    
    return total_zeros / total_elements

# ==============================
# PDP Layer Wrapper
# ==============================

class PDPLinear(nn.Module):
    """
    Linear layer with PDP applied.
    """
    def __init__(self, linear_layer, tau=0.02, target_sparsity=0.5):
        super(PDPLinear, self).__init__()
        self.linear = linear_layer
        self.tau = tau
        self.target_sparsity = target_sparsity
        self.register_buffer('mask', torch.ones_like(self.linear.weight.data))
        
    def forward(self, x):
        # Compute current threshold for target sparsity
        threshold = compute_threshold(self.linear.weight.data, self.target_sparsity)
        
        # Generate soft mask
        self.mask = pdp_soft_mask(self.linear.weight.data, threshold, self.tau)
        
        # Apply mask to weights
        masked_weight = self.linear.weight.data * self.mask
            
        # Save the masked weights temporarily for forward pass
        weight_original = self.linear.weight.data.clone()
        self.linear.weight.data = masked_weight
        
        # Perform forward pass
        output = self.linear(x)
        
        # Restore original weights
        self.linear.weight.data = weight_original
        
        return output

# ==============================
# Simple Test Function
# ==============================

def convert_gpt2_to_pdp(model, tau=0.02, target_sparsity=0.5):
    """Convert all linear layers to PDP layers."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Get parent module
            parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
            parent = model if parent_name == '' else getattr(model, parent_name.split('.')[0])
            
            for n in parent_name.split('.')[1:]:
                parent = getattr(parent, n)
            
            # Replace with PDP layer
            setattr(parent, child_name, PDPLinear(module, tau, target_sparsity))
    
    return model

def test_pdp(use_wandb=False, tau=0.02, improved_masking=False, beta=5.0, log_to_wandb=True):
    """Simple test function to verify PDP implementation."""
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project="pdp-gpt2-test",
            config={
                "tau": tau,
                "improved_masking": improved_masking,
                "beta": beta
            },
            name=f"pdp_tau{tau}_improved{improved_masking}"
        )
    
    print("Loading GPT-2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Count parameters before PDP
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters before PDP: {total_params:,}")
    
    # Apply PDP with increasing sparsity
    sparsity_levels = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is changing the way we live and work.",
        "Machine learning models can be compressed using various techniques.",
        "Neural networks with millions of parameters often contain redundancy."
    ]
    
    # Prepare table for wandb
    if use_wandb:
        wandb.Table(columns=["sparsity", "loss", "perplexity", "actual_sparsity", "sample_text", "generated_text"])
    
    results = []
    print("\nTesting PDP at different sparsity levels:")
    
    for sparsity in sparsity_levels:
        print(f"\nSparsity level: {sparsity:.2f}")
        
        # Apply PDP to model (with standard or improved masking)
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        model = convert_gpt2_to_pdp(model, tau=tau, target_sparsity=sparsity)
        
        avg_loss = 0
        avg_perplexity = 0
        generated_texts = []
        
        for sample_text in sample_texts:
            inputs = tokenizer(sample_text, return_tensors="pt").to(device)
            
            # Run forward pass
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                
            # Calculate loss and perplexity
            loss = outputs.loss.item()
            perplexity = math.exp(loss)
            
            avg_loss += loss
            avg_perplexity += perplexity
            
            # Generate some text
            input_ids = inputs["input_ids"]
            gen_tokens = model.generate(
                input_ids,
                max_length=50,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            generated_texts.append(gen_text)
        
        # Get actual sparsity
        actual_sparsity = get_sparsity(model)
        
        # Calculate averages
        avg_loss /= len(sample_texts)
        avg_perplexity /= len(sample_texts)
        
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Perplexity: {avg_perplexity:.4f}")
        print(f"Actual sparsity: {actual_sparsity:.4f}")
        print(f"Sample generated text: {generated_texts[0]}")
        
        # Collect results
        results.append({
            "sparsity": sparsity,
            "loss": avg_loss,
            "perplexity": avg_perplexity,
            "actual_sparsity": actual_sparsity,
            "generated_texts": generated_texts
        })
        
        # Log to wandb
        if use_wandb and log_to_wandb:
            wandb.log({
                "sparsity": sparsity,
                "loss": avg_loss,
                "perplexity": avg_perplexity,
                "actual_sparsity": actual_sparsity,
                "sparsity_gap": actual_sparsity - sparsity
            })
            
            # Add examples to table
            for i, (text, gen_text) in enumerate(zip(sample_texts, generated_texts)):
                wandb.log({
                    f"generation_{i}": wandb.Table(
                        columns=["sparsity", "input", "output"],
                        data=[[sparsity, text, gen_text]]
                    )
                })
    
    # Create summary plots in wandb
    if use_wandb:
        # Create custom chart
        sparsity_values = [r["sparsity"] for r in results]
        perplexity_values = [r["perplexity"] for r in results]
        actual_sparsity_values = [r["actual_sparsity"] for r in results]
        
        wandb.log({
            "sparsity_vs_perplexity": wandb.plot.line_series(
                xs=sparsity_values,
                ys=[perplexity_values],
                keys=["perplexity"],
                title="Sparsity vs. Perplexity",
                xname="Target Sparsity"
            ),
            "target_vs_actual_sparsity": wandb.plot.line_series(
                xs=sparsity_values,
                ys=[actual_sparsity_values, sparsity_values],
                keys=["actual", "target"],
                title="Target vs. Actual Sparsity",
                xname="Target Sparsity"
            )
        })
        
        wandb.finish()
    
    return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test GPT-2 with PDP pruning")
    
    # Basic parameters
    parser.add_argument("--tau", type=float, default=0.02, help="Temperature for soft masking")
    parser.add_argument("--improved_masking", action="store_true", help="Use improved soft masking")
    parser.add_argument("--beta", type=float, default=5.0, help="Beta parameter for improved masking")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test_pdp(
        use_wandb=args.use_wandb,
        tau=args.tau,
        improved_masking=args.improved_masking,
        beta=args.beta
    )