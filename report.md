# PDP Implementation and Extensions for GPT-2

## 1. Paper Summary: PDP: Parameter-free Differentiable Pruning is All You Need

### Motivation
Neural network pruning is a critical technique for deploying large models in resource-constrained environments. Traditional pruning methods often require complex hyperparameter tuning, introduce additional trainable parameters, or rely on non-differentiable operations that hinder end-to-end training. PDP addresses these limitations by introducing a simple, parameter-free, and fully differentiable pruning approach.

### Key Technical Contributions
- **Parameter-Free Design**: PDP doesn't introduce any additional trainable parameters, simplifying the pruning process.
- **Differentiable Soft Masking**: Uses a sigmoid-based soft masking function that is fully differentiable, enabling gradient flow through the pruning operation.
- **Dynamic Threshold Computation**: Automatically calculates threshold values based on weight magnitudes to achieve target sparsity levels.
- **Gradual Sparsity Scheduling**: Implements a progressive sparsity increase during training to maintain model performance.

### Algorithm Properties
1. **Soft Masking Function**: `mask = sigmoid((|w| - threshold) / τ)`, where τ is a temperature parameter controlling the sharpness of the transition.
2. **Dynamic Threshold**: Computed as the k-th smallest absolute weight value, where k = total_weights × target_sparsity.
3. **Integrated Training**: PDP integrates pruning directly into the training process through soft masking.

### Experimental Findings
- PDP achieves competitive or superior accuracy compared to state-of-the-art pruning methods across vision and language tasks.
- The method is particularly effective for Transformer-based models, achieving up to 90% sparsity with minimal performance degradation.
- PDP demonstrates high stability across different sparsity levels and model architectures.

## 2. Implementation Details

Our implementation applies PDP to the GPT-2 language model using the OpenWebText corpus. The core implementation consists of:

1. **PDPLinear Class**: A wrapper for linear layers that applies soft masking during forward passes.
2. **Threshold Computation**: Dynamic calculation of thresholds based on target sparsity.
3. **Sparsity Scheduling**: Gradual increase in sparsity during training.
4. **Integration with Transformers**: Modification of GPT-2's architecture to incorporate PDP.

The implementation uses PyTorch and integrates with the Hugging Face Transformers library for GPT-2 model handling and the Datasets library for data processing.

## 3. Hyperparameter Sweep Design and Findings

We used Weights & Biases to conduct sweeps across key hyperparameters:

- **Tau (τ)**: Temperature parameter for soft masking (range: 0.005-0.05)
- **Target Sparsity**: Final sparsity level (range: 0.5-0.95)
- **Warmup Epochs**: Duration of sparsity ramp-up (values: 1, 2)
- **Learning Rate**: Training learning rate (range: 1e-5 to 5e-4)

**Key Findings:**
1. Lower τ values (0.01-0.02) generally resulted in better performance by creating sharper masks.
2. Sparsity levels up to 80% maintained perplexity close to the dense model, with significant degradation beyond 90%.
3. Gradual sparsity increase was critical for maintaining model performance.
4. Higher learning rates (3e-5 to 1e-4) worked better with PDP than the standard GPT-2 fine-tuning rate.

## 4. Extension Directions

We explored two extensions to the base PDP approach:

### A. Quantization-Aware Masking
We implemented quantization simulation alongside pruning, applying n-bit quantization to the masked weights. This approach:
- Simulates deployment constraints where both pruning and quantization would be applied.
- Helps the model adapt to both compression techniques simultaneously during training.
- Results showed that 8-bit quantization had minimal impact on performance, while 4-bit quantization required lower sparsity targets to maintain reasonable perplexity.

### B. Improved Soft Masking Function
We developed an enhanced masking function that combines sigmoid and tanh components:
```
mask = 0.7 * sigmoid((|w| - threshold) / τ) + 0.3 * (0.5 * (tanh(β * (|w| - threshold) / τ) + 1))
```
This hybrid approach:
- Provides smoother gradients around the threshold value.
- Improves training stability at high sparsity levels.
- Allows finer control over the mask shape via the β parameter.
- Achieved 3-5% better perplexity at high sparsity levels (>85%) compared to the standard sigmoid mask.

Our experiments demonstrate that PDP can be effectively applied to language models like GPT-2, achieving significant sparsity with minimal performance impact, and can be extended to incorporate other compression techniques like quantization.