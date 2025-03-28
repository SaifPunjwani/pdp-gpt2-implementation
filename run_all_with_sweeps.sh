#!/bin/bash
# Master script to run both experiments and sweeps

# Default project name (can be changed below)
DEFAULT_PROJECT_NAME="pdp-gpt2-implementation"

# Ask user for project name
echo "Enter W&B project name (leave blank to use $DEFAULT_PROJECT_NAME):"
read custom_project
PROJECT_NAME=${custom_project:-$DEFAULT_PROJECT_NAME}
echo "Using project name: $PROJECT_NAME"

# Create necessary directories
mkdir -p logs
mkdir -p output
mkdir -p visualization_output

# =============================================
# Function to check if we're in a tmux session
# =============================================
in_tmux() {
    if [ -n "$TMUX" ]; then
        return 0  # True, we are in tmux
    else
        return 1  # False, not in tmux
    fi
}

# =============================================
# Check if running in tmux, if not, warn and offer to start
# =============================================
if ! in_tmux; then
    echo "WARNING: You are not running in a tmux session."
    echo "Running experiments without tmux means they will stop if you disconnect."
    echo ""
    read -p "Would you like to start a tmux session now? (y/n): " start_tmux
    
    if [[ "$start_tmux" == "y" || "$start_tmux" == "Y" ]]; then
        echo "Starting a new tmux session called pdp_master..."
        echo "This script will restart inside tmux."
        echo "Use Ctrl+b, d to detach without stopping experiments."
        sleep 2
        # Start a new tmux session and run this script inside it
        exec tmux new -s pdp_master "$0" "$@"
        exit
    else
        echo "Continuing without tmux. Be careful not to disconnect!"
    fi
fi

# =============================================
# Select what to run
# =============================================
echo "===== PDP GPT-2 Master Runner ====="
echo "Starting time: $(date)"
echo "Project: $PROJECT_NAME"
echo ""
echo "What would you like to run?"
echo "1. Run all experiments (30+ individual runs)"
echo "2. Run a hyperparameter sweep"
echo "3. Run both experiments and sweep"
echo "4. Generate visualizations from existing runs"
echo ""
read -p "Enter your choice (1-4): " choice

# =============================================
# Run option based on selection
# =============================================
case $choice in
    1)
        echo "Running all experiments..."
        bash ./run_all_experiments.sh "$PROJECT_NAME"
        ;;
    2)
        echo "Running hyperparameter sweep..."
        
        # Choose which sweep config to use
        echo ""
        echo "Which sweep configuration would you like to use?"
        echo "1. Full sweep (all parameters) - sweep_config.yaml"
        echo "2. Quick test sweep (few parameters) - sweep_config_test.yaml"
        echo "3. Simple sweep (visualizations only) - simple_sweep_config.yaml"
        echo ""
        read -p "Enter your choice (1-3): " sweep_choice
        
        case $sweep_choice in
            1) config_file="sweep_config.yaml" ;;
            2) config_file="sweep_config_test.yaml" ;;
            3) config_file="simple_sweep_config.yaml" ;;
            *) 
                echo "Invalid choice, using sweep_config.yaml"
                config_file="sweep_config.yaml"
                ;;
        esac
        
        echo "Initializing sweep with $config_file..."
        SWEEP_ID=$(wandb sweep $config_file | grep -oP '(?<=wandb agent )[^ ]+')
        
        if [ -z "$SWEEP_ID" ]; then
            echo "Failed to get sweep ID. Please check wandb login status and try again."
            exit 1
        fi
        
        echo "Starting sweep agent for sweep ID: $SWEEP_ID"
        read -p "How many sweep agents to run in parallel? (1-8): " agent_count
        
        # Validate input
        if ! [[ "$agent_count" =~ ^[1-8]$ ]]; then
            echo "Invalid input, using 1 agent"
            agent_count=1
        fi
        
        # Start agents
        for i in $(seq 1 $agent_count); do
            if [ $i -eq 1 ]; then
                # First agent in current window
                echo "Starting agent 1..."
                wandb agent $SWEEP_ID
            else
                # Additional agents in new tmux windows
                echo "Starting agent $i in a new window..."
                tmux new-window -t pdp_master:$i -n "agent$i" "wandb agent $SWEEP_ID"
            fi
        done
        ;;
    3)
        echo "Running experiments first, then sweep..."
        
        # Run experiments
        bash ./run_all_experiments.sh "$PROJECT_NAME"
        
        # Then run sweep
        echo "Initializing sweep with sweep_config.yaml..."
        SWEEP_ID=$(wandb sweep sweep_config.yaml | grep -oP '(?<=wandb agent )[^ ]+')
        
        if [ -z "$SWEEP_ID" ]; then
            echo "Failed to get sweep ID. Experiments ran successfully, but sweep failed to start."
            exit 1
        fi
        
        echo "Starting sweep agent for sweep ID: $SWEEP_ID"
        wandb agent $SWEEP_ID
        ;;
    4)
        echo "Generating visualizations from existing W&B runs..."
        python enhanced_visualization.py --project $PROJECT_NAME --output_dir visualization_output
        echo "Visualizations saved to visualization_output/"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "===== Process completed ====="
echo "End time: $(date)"

if in_tmux; then
    echo ""
    echo "You are in a tmux session. To detach without stopping processes, press Ctrl+b, d"
    echo "To reconnect later, use: tmux attach -t pdp_master"
fi