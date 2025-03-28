#!/bin/bash
# Comprehensive experiment runner for PDP hyperparameter exploration

# Create output directories
mkdir -p logs
mkdir -p output
mkdir -p visualization_output

# Set the project name
PROJECT_NAME="pdp-gpt2-implementation"

echo "===== PDP GPT-2 Comprehensive Experiments ====="
echo "Starting time: $(date)"
echo "Project: $PROJECT_NAME"

# Helper function to log experiments
log_start() {
    echo "$(date): Starting experiment: $1" | tee -a logs/experiment_log.txt
}

log_end() {
    echo "$(date): Finished experiment: $1" | tee -a logs/experiment_log.txt
    echo "-----------------------------------------" | tee -a logs/experiment_log.txt
}

# Make sure wandb is logged in
echo "Checking wandb login status..."
wandb status

# ========================
# Standard PDP Experiments
# ========================

# 1. Standard masking with different tau values
for tau in 0.01 0.02 0.05; do
    for sparsity in 0.5 0.7 0.8 0.9; do
        run_name="standard_tau${tau}_sparsity${sparsity}"
        log_start "$run_name"
        
        echo "Running standard PDP with tau=$tau, sparsity=$sparsity"
        python main.py \
            --tau $tau \
            --sparsity $sparsity \
            --num_epochs 3 \
            --batch_size 16 \
            --warmup_epochs 1 \
            --use_wandb \
            --run_name "$run_name" \
            --output_dir "output/$run_name" 2>&1 | tee logs/$run_name.log
            
        log_end "$run_name"
    done
done

# ========================
# Improved Masking Experiments
# ========================

# 2. Improved masking with different beta values
for beta in 2.0 5.0 10.0; do
    for sparsity in 0.7 0.8 0.9; do
        run_name="improved_tau0.02_beta${beta}_sparsity${sparsity}"
        log_start "$run_name"
        
        echo "Running improved PDP with tau=0.02, beta=$beta, sparsity=$sparsity"
        python main.py \
            --tau 0.02 \
            --sparsity $sparsity \
            --improved_masking \
            --beta $beta \
            --num_epochs 3 \
            --batch_size 16 \
            --warmup_epochs 1 \
            --use_wandb \
            --run_name "$run_name" \
            --output_dir "output/$run_name" 2>&1 | tee logs/$run_name.log
            
        log_end "$run_name"
    done
done

# ========================
# Quantization Experiments
# ========================

# 3. Quantization-aware masking
for bits in 4 8; do
    for sparsity in 0.5 0.7 0.8; do
        run_name="quantized_bits${bits}_sparsity${sparsity}"
        log_start "$run_name"
        
        echo "Running quantized PDP with bits=$bits, sparsity=$sparsity"
        python main.py \
            --tau 0.02 \
            --sparsity $sparsity \
            --quantize \
            --bits $bits \
            --num_epochs 3 \
            --batch_size 16 \
            --warmup_epochs 1 \
            --use_wandb \
            --run_name "$run_name" \
            --output_dir "output/$run_name" 2>&1 | tee logs/$run_name.log
            
        log_end "$run_name"
    done
done

# ========================
# Warmup Experiments
# ========================

# 4. Different warmup durations
for warmup in 0 1 2; do
    run_name="warmup_epochs${warmup}_sparsity0.8"
    log_start "$run_name"
    
    echo "Running PDP with warmup_epochs=$warmup, sparsity=0.8"
    python main.py \
        --tau 0.02 \
        --sparsity 0.8 \
        --warmup_epochs $warmup \
        --num_epochs 3 \
        --batch_size 16 \
        --use_wandb \
        --run_name "$run_name" \
        --output_dir "output/$run_name" 2>&1 | tee logs/$run_name.log
        
    log_end "$run_name"
done

# ========================
# Custom Optimizer Experiments
# ========================

# 5. Test custom optimizer
for sparsity in 0.7 0.9; do
    run_name="custom_optimizer_sparsity${sparsity}"
    log_start "$run_name"
    
    echo "Running PDP with custom optimizer, sparsity=$sparsity"
    python main.py \
        --tau 0.02 \
        --sparsity $sparsity \
        --custom_optimizer \
        --num_epochs 3 \
        --batch_size 16 \
        --warmup_epochs 1 \
        --use_wandb \
        --run_name "$run_name" \
        --output_dir "output/$run_name" 2>&1 | tee logs/$run_name.log
        
    log_end "$run_name"
done

# ========================
# Generate Visualizations
# ========================

echo "Generating comprehensive visualizations..."
python enhanced_visualization.py --project $PROJECT_NAME --output_dir visualization_output

echo "===== All experiments completed ====="
echo "End time: $(date)"
echo "Check the 'visualization_output' directory for result visualizations"