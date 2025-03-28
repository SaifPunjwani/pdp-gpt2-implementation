#!/usr/bin/env python3
"""
Weights & Biases sweep agent for PDP
"""
import os
import wandb
import argparse
from test_pdp import test_pdp

def main():
    parser = argparse.ArgumentParser(description="Run W&B sweep agent for PDP")
    parser.add_argument("--sweep_id", type=str, required=True, help="W&B sweep ID")
    parser.add_argument("--project", type=str, default="pdp-gpt2-sweep", help="W&B project name")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity (username or team name)")
    args = parser.parse_args()
    
    def sweep_train():
        """Function that will be called by the sweep agent"""
        # Initialize a W&B run
        wandb.init()
        
        # Get hyperparameters from W&B
        config = wandb.config
        
        # Only use beta if improved_masking is enabled
        beta = config.beta if hasattr(config, 'beta') and config.improved_masking else 5.0
        
        # Update the run name to reflect actual parameters being used
        if wandb.run is not None:
            if config.improved_masking:
                wandb.run.name = f"tau{config.tau}_improved_beta{beta}"
            else:
                wandb.run.name = f"tau{config.tau}_standard"
        
        # Run test_pdp with the sweep parameters
        test_pdp(
            use_wandb=True,
            tau=config.tau,
            improved_masking=config.improved_masking,
            beta=beta,
            log_to_wandb=True
        )
    
    # Start the sweep agent
    wandb.agent(
        args.sweep_id,
        function=sweep_train,
        project=args.project,
        entity=args.entity
    )

if __name__ == "__main__":
    main()