import torch
import numpy as np
from transformers import PretrainedConfig, PreTrainedModel
from typing import List, Dict, Any, Tuple

def analyze_layer_importance(
    model: PreTrainedModel, 
    strategy: str = 'middle_layers'
) -> List[int]:
    """
    Analyzes the importance of layers to identify which ones to replace.

    The actual PostNAS methodology uses a sophisticated, task-dependent analysis.
    This function serves as a placeholder to simulate that process with common strategies.

    Args:
        model (PreTrainedModel): The pre-trained Hugging Face model to analyze.
        strategy (str): The strategy to use for identifying non-critical layers.
                        - 'first_half': Replaces the first half of the layers.
                        - 'alternate': Replaces every other layer.
                        - 'middle_layers': A common finding is that middle and late
                                           layers are more critical. This strategy
                                           preserves them and suggests replacing
                                           earlier layers.

    Returns:
        List[int]: A list of layer indices suggested for replacement with JetBlocks.
    """
    num_layers = model.config.num_hidden_layers
    
    if strategy == 'first_half':
        # Replace the first 50% of the layers.
        layers_to_replace = list(range(num_layers // 2))
    elif strategy == 'alternate':
        # Replace every other layer, starting from the second layer.
        layers_to_replace = list(range(1, num_layers, 2))
    elif strategy == 'middle_layers':
        # Preserve the middle and final layers, which are often critical for
        # complex reasoning tasks. Replace a subset of the early layers.
        # For example, preserve the last 50% and replace a quarter of the first 50%.
        critical_start_index = num_layers // 2
        non_critical_layers = list(range(critical_start_index))
        # Select a subset of non-critical layers to replace.
        layers_to_replace = non_critical_layers[::4] # Replace every 4th layer in the non-critical section
    else:
        raise ValueError(f"Unknown layer importance strategy: {strategy}")

    print(f"Layer Importance Analysis ({strategy}): Suggesting replacement of layers {layers_to_replace}")
    return layers_to_replace


def calculate_kv_cache_size(
    config: PretrainedConfig,
    batch_size: int,
    sequence_length: int,
    precision_bytes: int = 2 # FP16/BF16
) -> float:
    """
    Calculates the theoretical KV cache size for a given model configuration.

    This is a critical hardware-aware metric for evaluating architectures, as
    highlighted in the Jet-Nemotron paper.

    Args:
        config (PretrainedConfig): The model's configuration object.
        batch_size (int): The batch size used for inference.
        sequence_length (int): The context length of the input.
        precision_bytes (int): The number of bytes per parameter (e.g., 2 for FP16, 4 for FP32).

    Returns:
        float: The total KV cache size in megabytes (MB).
    """
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    # Each token requires storing a Key and a Value vector for each layer.
    # Size = batch_size * seq_len * num_layers * (key_size + value_size) * precision
    key_cache_per_layer = batch_size * sequence_length * num_heads * head_dim
    value_cache_per_layer = batch_size * sequence_length * num_heads * head_dim
    
    total_cache_bytes = num_layers * (key_cache_per_layer + value_cache_per_layer) * precision_bytes
    
    # Convert bytes to megabytes
    total_cache_mb = total_cache_bytes / (1024 * 1024)
    
    return total_cache_mb


def get_hardware_aware_search_space() -> List[Dict[str, Any]]:
    """
    Defines a search space for hardware-aware architecture search.

    This function returns a list of hyperparameter configurations inspired by
    Table 2 in the provided research paper images. The goal is to find a
    trade-off between accuracy and hardware efficiency (throughput, cache size).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                              represents a model architecture configuration.
    """
    # d_K, d_V, n_head configurations from Table 2
    search_space = [
        # n_head = 4
        {'d_k': 256, 'd_v': 288, 'n_head': 4},
        {'d_k': 192, 'd_v': 384, 'n_head': 4}, # Original design in gray
        {'d_k': 128, 'd_v': 576, 'n_head': 4},
        # n_head = 8
        {'d_k': 256, 'd_v': 144, 'n_head': 8},
        {'d_k': 192, 'd_v': 192, 'n_head': 8},
        {'d_k': 128, 'd_v': 288, 'n_head': 8},
        # n_head = 12
        {'d_k': 128, 'd_v': 192, 'n_head': 12},
        {'d_k': 96, 'd_v': 256, 'n_head': 12}, # New design in blue
        {'d_k': 64, 'd_v': 384, 'n_head': 12},
    ]
    return search_space


def update_model_config(
    base_config: PretrainedConfig, 
    architecture_params: Dict[str, Any]
) -> PretrainedConfig:
    """
    Updates a model's configuration with new architectural hyperparameters.

    Args:
        base_config (PretrainedConfig): The original configuration of the pre-trained model.
        architecture_params (Dict[str, Any]): A dictionary containing the new
                                               hyperparameters to apply.

    Returns:
        PretrainedConfig: A new configuration object with updated values.
    """
    new_config = base_config.from_dict(base_config.to_dict()) # Deep copy

    new_config.num_attention_heads = architecture_params['n_head']
    
    # The total hidden size must be consistent. In many transformer architectures,
    # hidden_size = num_heads * head_dim. We assume d_k = d_v = head_dim.
    # The paper's table suggests d_k and d_v can differ, which implies a more
    # complex attention head. For a standard implementation, we enforce d_k = d_v.
    # We will prioritize d_k for the head dimension calculation.
    head_dim = architecture_params['d_k']
    new_config.hidden_size = new_config.num_attention_heads * head_dim

    print(f"Updated model config: heads={new_config.num_attention_heads}, head_dim={head_dim}, hidden_size={new_config.hidden_size}")

    return new_config
