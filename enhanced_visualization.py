#!/usr/bin/env python3
"""
Enhanced visualization for PDP experiments
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import argparse
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick

# Set the style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

def fetch_runs(project_name="pdp-gpt2-implementation", entity=None):
    """
    Fetch runs from W&B
    """
    # Initialize wandb API
    api = wandb.Api()
    
    # Get all runs for the project
    runs = api.runs(f"{entity}/{project_name}" if entity else project_name)
    
    # Process the runs data
    data = []
    for run in runs:
        # Skip runs that are not finished or have no summary
        if run.state != "finished" or not hasattr(run, 'summary'):
            continue
            
        # Extract hyperparameters
        config = run.config
        tau = config.get('tau', None)
        sparsity = config.get('sparsity', None)
        improved_masking = config.get('improved_masking', False)
        beta = config.get('beta', None) if improved_masking else None
        warmup_epochs = config.get('warmup_epochs', None)
        
        # Extract metrics
        metrics = {}
        if hasattr(run.summary, '_json_dict'):
            metrics = {
                'perplexity': run.summary._json_dict.get('perplexity', None),
                'eval_loss': run.summary._json_dict.get('eval_loss', None),
                'final_sparsity': run.summary._json_dict.get('final_sparsity', None),
            }
        
        # Append data
        data.append({
            'run_id': run.id,
            'name': run.name,
            'tau': tau,
            'sparsity': sparsity,
            'improved_masking': improved_masking,
            'beta': beta,
            'warmup_epochs': warmup_epochs,
            **metrics
        })
    
    return pd.DataFrame(data)

def create_sparsity_perplexity_plot(df, output_dir="visualization_output"):
    """Create sparsity vs. perplexity plot"""
    plt.figure(figsize=(10, 6))
    
    # Group by improved_masking for separate lines
    improved = df[df['improved_masking'] == True]
    standard = df[df['improved_masking'] == False]
    
    # Plot lines for standard masking
    if not standard.empty:
        sns.lineplot(data=standard, x='sparsity', y='perplexity', marker='o', 
                    label='Standard Masking', linewidth=2)
    
    # Plot lines for improved masking
    if not improved.empty:
        sns.lineplot(data=improved, x='sparsity', y='perplexity', marker='s',
                    label='Improved Masking', linewidth=2, linestyle='--')
    
    plt.xlabel('Target Sparsity', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title('Impact of Sparsity on Model Perplexity', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Formatting
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/sparsity_vs_perplexity.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_tau_heatmap(df, output_dir="visualization_output"):
    """Create heatmap of tau vs. sparsity on perplexity"""
    if 'tau' not in df.columns or df['tau'].isnull().all() or len(df['tau'].unique()) <= 1:
        print("Not enough tau values for heatmap")
        return
        
    # Pivot the data for the heatmap
    pivot = df.pivot_table(
        values='perplexity', 
        index='tau', 
        columns='sparsity', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis", 
                    cbar_kws={'label': 'Perplexity'})
    
    # Format column labels as percentages
    ax.set_xticklabels([f"{x:.0%}" for x in pivot.columns])
    
    plt.title('Temperature (τ) vs. Sparsity Effect on Perplexity', fontsize=14)
    plt.xlabel('Target Sparsity', fontsize=12)
    plt.ylabel('Temperature (τ)', fontsize=12)
    
    # Save figure
    plt.savefig(f"{output_dir}/tau_sparsity_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_masking_comparison_plot(df, output_dir="visualization_output"):
    """Create comparison of standard vs. improved masking"""
    # Filter data
    mask_comparison = df[df['sparsity'] > 0.5]  # Focus on higher sparsity levels
    
    if 'improved_masking' not in df.columns or len(df['improved_masking'].unique()) <= 1:
        print("Not enough masking types for comparison")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Group by masking type
    sns.barplot(
        data=mask_comparison, 
        x='sparsity', 
        y='perplexity', 
        hue='improved_masking', 
        palette=["#3498db", "#e74c3c"]
    )
    
    # Get current handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # Replace boolean labels with descriptive text
    labels = ['Standard Masking' if label == 'False' else 'Improved Masking' for label in labels]
    # Update legend
    plt.legend(handles, labels, title="Masking Type", fontsize=10)
    
    plt.xlabel('Target Sparsity', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title('Standard vs. Improved Masking Performance', fontsize=14)
    
    # Format x-axis as percentages
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Save figure
    plt.savefig(f"{output_dir}/masking_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_dashboard(df, output_dir="visualization_output"):
    """Create a comprehensive dashboard of visualizations"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a large figure for the dashboard
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig)
    
    # Filter out rows with missing perplexity
    df_valid = df.dropna(subset=['perplexity'])
    
    # 1. Sparsity vs. Perplexity line plot
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create grouped data by masking type
    improved = df_valid[df_valid['improved_masking'] == True]
    standard = df_valid[df_valid['improved_masking'] == False]
    
    # Plot lines for standard masking
    if not standard.empty:
        sns.lineplot(data=standard, x='sparsity', y='perplexity', marker='o', 
                    label='Standard Masking', linewidth=2, ax=ax1)
    
    # Plot lines for improved masking
    if not improved.empty:
        sns.lineplot(data=improved, x='sparsity', y='perplexity', marker='s',
                    label='Improved Masking', linewidth=2, linestyle='--', ax=ax1)
    
    ax1.set_xlabel('Target Sparsity', fontsize=12)
    ax1.set_ylabel('Perplexity', fontsize=12)
    ax1.set_title('Impact of Sparsity on Model Perplexity', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.legend(fontsize=10)
    
    # 2. Temperature impact on perplexity
    ax2 = fig.add_subplot(gs[0, 1])
    if 'tau' in df_valid.columns and not df_valid['tau'].isnull().all():
        sns.boxplot(data=df_valid, x='tau', y='perplexity', ax=ax2)
        ax2.set_xlabel('Temperature (τ)', fontsize=12)
        ax2.set_ylabel('Perplexity', fontsize=12)
        ax2.set_title('Effect of Temperature on Perplexity', fontsize=14)
    
    # 3. Masking Type Comparison (Standard vs. Improved)
    ax3 = fig.add_subplot(gs[1, 0])
    if 'improved_masking' in df_valid.columns and len(df_valid['improved_masking'].unique()) > 1:
        # Create masking type column for display
        df_valid['masking_type'] = df_valid['improved_masking'].apply(
            lambda x: 'Improved Masking' if x else 'Standard Masking'
        )
        
        sns.boxplot(data=df_valid, x='masking_type', y='perplexity', ax=ax3)
        ax3.set_xlabel('Masking Type', fontsize=12)
        ax3.set_ylabel('Perplexity', fontsize=12)
        ax3.set_title('Standard vs. Improved Masking Performance', fontsize=14)
    
    # 4. Warmup Impact (if data is available)
    ax4 = fig.add_subplot(gs[1, 1])
    if 'warmup_epochs' in df_valid.columns and not df_valid['warmup_epochs'].isnull().all():
        sns.boxplot(data=df_valid, x='warmup_epochs', y='perplexity', ax=ax4)
        ax4.set_xlabel('Warmup Epochs', fontsize=12)
        ax4.set_ylabel('Perplexity', fontsize=12)
        ax4.set_title('Impact of Warmup Duration on Perplexity', fontsize=14)
    
    # 5. Target vs. Actual Sparsity
    ax5 = fig.add_subplot(gs[2, 0])
    if 'final_sparsity' in df_valid.columns and not df_valid['final_sparsity'].isnull().all():
        # Create a scatter plot with target vs actual
        ax5.scatter(df_valid['sparsity'], df_valid['final_sparsity'], 
                   alpha=0.7, c=df_valid['perplexity'], cmap='viridis')
        
        # Add a diagonal reference line (perfect match)
        lims = [
            np.min([ax5.get_xlim(), ax5.get_ylim()]),
            np.max([ax5.get_xlim(), ax5.get_ylim()]),
        ]
        ax5.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        
        ax5.set_xlabel('Target Sparsity', fontsize=12)
        ax5.set_ylabel('Actual Sparsity', fontsize=12)
        ax5.set_title('Target vs. Achieved Sparsity', fontsize=14)
        ax5.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax5.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # 6. Improved Masking Beta Parameter (if data is available)
    ax6 = fig.add_subplot(gs[2, 1])
    improved_with_beta = df_valid[(df_valid['improved_masking'] == True) & (~df_valid['beta'].isnull())]
    if not improved_with_beta.empty and len(improved_with_beta['beta'].unique()) > 1:
        sns.boxplot(data=improved_with_beta, x='beta', y='perplexity', ax=ax6)
        ax6.set_xlabel('Beta Parameter (β)', fontsize=12)
        ax6.set_ylabel('Perplexity', fontsize=12)
        ax6.set_title('Impact of Beta Parameter on Improved Masking', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pdp_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dashboard saved to {output_dir}/pdp_dashboard.png")
    
    return fig

def main():
    """Main function to run visualization"""
    parser = argparse.ArgumentParser(description="Generate enhanced visualizations for PDP experiments")
    parser.add_argument("--project", type=str, default="pdp-gpt2-implementation",
                        help="W&B project name")
    parser.add_argument("--entity", type=str, default=None,
                        help="W&B entity (username or team name)")
    parser.add_argument("--output_dir", type=str, default="visualization_output",
                        help="Directory to save visualizations")
    args = parser.parse_args()
    
    # Fetch runs data
    print(f"Fetching runs from W&B project: {args.project}")
    df = fetch_runs(args.project, args.entity)
    
    if df.empty:
        print("No runs found or no finished runs with metrics.")
        return
    
    print(f"Found {len(df)} valid runs with metrics.")
    
    # Create individual plots
    create_sparsity_perplexity_plot(df, args.output_dir)
    create_tau_heatmap(df, args.output_dir)
    create_masking_comparison_plot(df, args.output_dir)
    
    # Create comprehensive dashboard
    create_dashboard(df, args.output_dir)
    
    print(f"All visualizations saved to {args.output_dir}/")

if __name__ == "__main__":
    main()