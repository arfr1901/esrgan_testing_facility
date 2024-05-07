def model_path_of(model_name):
    return "C:/Users/darks/AppData/Local/Android/Sdk/platform-tools/" + model_name


# Define optimization strategies
quantization_strategies = [
    "drq",             # Baseline, uses Dynamic Range Quantization
    "f16q",            # Float 16 Quantization
    "fiq",             # Full Integer Quantization
    # "Custom1",         # Custom Quantization Configuration 1
    # "Custom2"          # Custom Quantization Configuration 2
]

pruning_strategies = [
    "UNSTRUCTURED",    # Unstructured pruning (weight sparsification)
    "STRUCTURED",      # Structured pruning (channels, filters, neurons)
    "MAGNITUDE",       # Magnitude-based pruning
    "L0_REG",          # L0 Regularization pruning
    "LAYER_PRUNING",   # Layer pruning based on relevance
    "DYNAMIC",         # Dynamic pruning (during inference)
    "SLIMMING",        # Network slimming via batch norm
    "ITERATIVE"        # Iterative pruning and fine-tuning
]

clustering_strategies = [
    "KMEANS",          # K-Means clustering
    "VQ",              # Vector Quantization
    "PQ",              # Product Quantization
    "RQ",              # Residual Quantization
    "BINARY_NETS",     # Binary Networks
    "TERNARY_NETS"     # Ternary Networks
]

mixed_optimization_strategies = [
    ""
]

custom_optimization_strategies = [
    ""
]
