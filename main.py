import os
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import get_scheduler
import wandb
from tqdm import tqdm

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
# Quantization-Aware Masking (Extension)
# ==============================

def quantize_weights(weights, bits=8):
    """
    Simulate quantization for the masked weights.
    
    Args:
        weights (torch.Tensor): Weight tensor
        bits (int): Number of bits for quantization
        
    Returns:
        torch.Tensor: Quantized weights
    """
    # Calculate range
    min_val = weights.min()
    max_val = weights.max()
    
    # Calculate step size
    step_size = (max_val - min_val) / (2**bits - 1)
    
    # Quantize
    weights_q = torch.round((weights - min_val) / step_size) * step_size + min_val
    
    return weights_q

# ==============================
# Improved Soft Masking (Extension)
# ==============================

def improved_soft_mask(weight, threshold, tau, beta=5.0):
    """
    Enhanced soft masking function with smoother gradient transition.
    
    Args:
        weight (torch.Tensor): The weight tensor
        threshold (float): Pruning threshold (per-layer, per-step)
        tau (float): Temperature controlling sharpness of transition
        beta (float): Scaling factor for the tanh component
        
    Returns:
        mask (torch.Tensor): Values in [0, 1] representing soft mask
    """
    # Combination of sigmoid and scaled tanh for improved gradient flow
    sig = torch.sigmoid((weight.abs() - threshold) / tau)
    tanh_comp = 0.5 * (torch.tanh(beta * (weight.abs() - threshold) / tau) + 1)
    
    # Weighted combination
    return 0.7 * sig + 0.3 * tanh_comp

# ==============================
# Custom Optimizer for PDP (Extension)
# ==============================

class PDPOptimizer(torch.optim.Optimizer):
    """
    Custom optimizer for PDP that adjusts learning rates based on mask values.
    """
    def __init__(self, params, base_optimizer, tau=0.02, sparsity=0.9, lr=1e-3, weight_decay=0):
        self.base_optimizer = base_optimizer
        self.tau = tau
        self.sparsity = sparsity
        
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(PDPOptimizer, self).__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                if hasattr(p, 'mask'):
                    # Scale gradients based on mask values to improve training stability
                    adjusted_grad = grad * (0.1 + 0.9 * p.mask)
                    p.grad.data = adjusted_grad
        
        # Let the base optimizer handle the actual parameter updates
        self.base_optimizer.step()
        
        return loss

# ==============================
# PDP Layer Wrapper
# ==============================

class PDPLinear(nn.Module):
    """
    Linear layer with PDP applied.
    """
    def __init__(self, linear_layer, tau=0.02, target_sparsity=0.5, 
                 quantize=False, bits=8, improved_masking=False, beta=5.0):
        super(PDPLinear, self).__init__()
        self.linear = linear_layer
        self.tau = tau
        self.target_sparsity = target_sparsity
        self.quantize = quantize
        self.bits = bits
        self.improved_masking = improved_masking
        self.beta = beta
        self.register_buffer('mask', torch.ones_like(self.linear.weight.data))
        
    def forward(self, x):
        # Compute current threshold for target sparsity
        threshold = compute_threshold(self.linear.weight.data, self.target_sparsity)
        
        # Generate soft mask
        if self.improved_masking:
            self.mask = improved_soft_mask(self.linear.weight.data, threshold, self.tau, self.beta)
        else:
            self.mask = pdp_soft_mask(self.linear.weight.data, threshold, self.tau)
        
        # Apply mask to weights
        masked_weight = self.linear.weight.data * self.mask
        
        # Apply quantization if enabled
        if self.quantize:
            masked_weight = quantize_weights(masked_weight, self.bits)
            
        # Save the masked weights temporarily for forward pass
        weight_original = self.linear.weight.data.clone()
        self.linear.weight.data = masked_weight
        
        # Perform forward pass
        output = self.linear(x)
        
        # Restore original weights
        self.linear.weight.data = weight_original
        
        return output

# ==============================
# Model Conversion Function
# ==============================

def convert_gpt2_to_pdp(model, tau=0.02, target_sparsity=0.5, 
                       quantize=False, bits=8, improved_masking=False, beta=5.0):
    """
    Convert a GPT-2 model to use PDP by replacing linear layers.
    
    Args:
        model: GPT-2 model
        tau: Temperature for soft masking
        target_sparsity: Target sparsity level
        quantize: Whether to enable quantization
        bits: Number of bits for quantization
        improved_masking: Whether to use improved masking function
        beta: Beta parameter for improved masking
        
    Returns:
        Modified model with PDP applied
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Get parent module
            parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
            parent = model if parent_name == '' else getattr(model, parent_name.split('.')[0])
            
            for n in parent_name.split('.')[1:]:
                parent = getattr(parent, n)
            
            # Replace with PDP layer
            setattr(parent, child_name, 
                   PDPLinear(module, tau, target_sparsity, 
                            quantize, bits, improved_masking, beta))
    
    return model

# ==============================
# Data Processing
# ==============================

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize a batch of text examples."""
    return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

class OpenWebTextDataset(torch.utils.data.Dataset):
    """Dataset wrapper for tokenized HF dataset to ensure PyTorch DataLoader compatibility."""
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset
        
    def __len__(self):
        return len(self.tokenized_dataset)
    
    def __getitem__(self, idx):
        item = self.tokenized_dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"])
        }

def prepare_dataset(tokenizer, batch_size=16, max_length=512):
    """Prepare and tokenize the OpenWebText dataset."""
    # Load a subset of OpenWebText for faster experimentation
    print("Loading OpenWebText dataset...")
    dataset = load_dataset("openwebtext", split="train[:0.1%]")
    
    # Tokenize the dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["text"]
    )
    
    # Split into train and validation
    tokenized_splits = tokenized_dataset.train_test_split(test_size=0.1)
    
    # Create PyTorch datasets
    train_dataset = OpenWebTextDataset(tokenized_splits["train"])
    eval_dataset = OpenWebTextDataset(tokenized_splits["test"])
    
    # Create data loaders
    print(f"Creating DataLoaders with batch size {batch_size}...")
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=batch_size
    )
    
    return train_dataloader, eval_dataloader

# ==============================
# Training Functions
# ==============================

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, args):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        args: Training arguments
        
    Returns:
        Average loss
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    sparsity_schedule = linear_sparsity_schedule(
        start_sparsity=0.0,
        target_sparsity=args.sparsity,
        start_step=epoch * len(dataloader),
        end_step=(epoch + args.warmup_epochs) * len(dataloader),
        current_step=epoch * len(dataloader)
    )
    
    for step, batch in enumerate(progress_bar):
        # Update target sparsity according to schedule
        current_step = epoch * len(dataloader) + step
        current_sparsity = linear_sparsity_schedule(
            start_sparsity=0.0,
            target_sparsity=args.sparsity,
            start_step=0,
            end_step=args.warmup_epochs * len(dataloader),
            current_step=current_step
        )
        
        # Update sparsity for all PDPLinear layers
        for module in model.modules():
            if isinstance(module, PDPLinear):
                module.target_sparsity = current_sparsity
        
        batch = {k: v.to(device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Update progress
        total_loss += loss.item()
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}", 
            "sparsity": f"{current_sparsity:.2f}",
            "lr": f"{scheduler.get_last_lr()[0]:.6f}"
        })
        
        # Log to wandb
        if args.use_wandb and step % args.log_interval == 0:
            model_sparsity = get_sparsity(model)
            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "target_sparsity": current_sparsity,
                "actual_sparsity": model_sparsity,
                "epoch": epoch,
                "step": current_step
            })
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, epoch, args):
    """
    Evaluate the model.
    
    Args:
        model: The model to evaluate
        dataloader: Evaluation data loader
        device: Device to evaluate on
        epoch: Current epoch number
        args: Training arguments
        
    Returns:
        Perplexity
    """
    model.eval()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Eval]")
    
    with torch.no_grad():
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    
    if args.use_wandb:
        wandb.log({
            "eval_loss": avg_loss,
            "perplexity": perplexity,
            "epoch": epoch
        })
    
    return perplexity

# ==============================
# Sparsity Scheduling
# ==============================

def linear_sparsity_schedule(start_sparsity, target_sparsity, start_step, end_step, current_step):
    """
    Linear scheduling of sparsity rate.
    
    Args:
        start_sparsity: Initial sparsity
        target_sparsity: Target sparsity
        start_step: Step to start increasing sparsity
        end_step: Step to reach target sparsity
        current_step: Current step
        
    Returns:
        Current target sparsity
    """
    if current_step <= start_step:
        return start_sparsity
    
    if current_step >= end_step:
        return target_sparsity
    
    # Linear interpolation
    step_ratio = (current_step - start_step) / (end_step - start_step)
    return start_sparsity + step_ratio * (target_sparsity - start_sparsity)

# ==============================
# Main Training Function
# ==============================

def train(args):
    """
    Main training function.
    
    Args:
        args: Training arguments
    """
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project="pdp-gpt2-implementation",
            config=vars(args),
            name=args.run_name
        )
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    
    # Apply PDP to model
    model = convert_gpt2_to_pdp(
        model, 
        tau=args.tau, 
        target_sparsity=0.0,  # Start at 0, will be scheduled
        quantize=args.quantize,
        bits=args.bits,
        improved_masking=args.improved_masking,
        beta=args.beta
    )
    
    model.to(device)
    
    # Prepare dataset
    train_dataloader, eval_dataloader = prepare_dataset(
        tokenizer, 
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Set up optimizer
    if args.custom_optimizer:
        base_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        optimizer = PDPOptimizer(
            model.parameters(),
            base_optimizer,
            tau=args.tau,
            sparsity=args.sparsity,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    
    # Set up learning rate scheduler
    num_training_steps = args.num_epochs * len(train_dataloader)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    best_perplexity = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        avg_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device, epoch, args)
        print(f"Average training loss: {avg_loss:.4f}")
        
        # Evaluate
        perplexity = evaluate(model, eval_dataloader, device, epoch, args)
        print(f"Perplexity: {perplexity:.4f}")
        
        # Save best model
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                model.save_pretrained(os.path.join(args.output_dir, "best_model"))
                print(f"Saved best model to {args.output_dir}/best_model")
    
    # Calculate final sparsity
    final_sparsity = get_sparsity(model)
    print(f"Final model sparsity: {final_sparsity:.4f}")
    
    # Close wandb
    if args.use_wandb:
        wandb.finish()
    
    return model, best_perplexity, final_sparsity

# ==============================
# Main
# ==============================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GPT-2 with PDP pruning")
    
    # Basic training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Learning rate warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    
    # PDP parameters
    parser.add_argument("--tau", type=float, default=0.02, help="Temperature for soft masking")
    parser.add_argument("--sparsity", type=float, default=0.9, help="Target sparsity level")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="Epochs to reach target sparsity")
    
    # Extension options
    parser.add_argument("--quantize", action="store_true", help="Enable quantization masking")
    parser.add_argument("--bits", type=int, default=8, help="Bits for quantization")
    parser.add_argument("--improved_masking", action="store_true", help="Use improved soft masking")
    parser.add_argument("--beta", type=float, default=5.0, help="Beta parameter for improved masking")
    parser.add_argument("--custom_optimizer", action="store_true", help="Use custom PDP optimizer")
    
    # Logging and output
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval in steps")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--run_name", type=str, default=None, help="Name for W&B run")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Default run name if not specified
    if args.run_name is None:
        extensions = []
        if args.quantize:
            extensions.append(f"quant{args.bits}")
        if args.improved_masking:
            extensions.append(f"impMask_b{args.beta}")
        if args.custom_optimizer:
            extensions.append("custOpt")
        
        ext_str = "_".join(extensions)
        args.run_name = f"pdp_s{args.sparsity}_t{args.tau}" + (f"_{ext_str}" if ext_str else "")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    model, perplexity, sparsity = train(args)
    print(f"Training complete! Final perplexity: {perplexity:.4f}, Sparsity: {sparsity:.4f}")