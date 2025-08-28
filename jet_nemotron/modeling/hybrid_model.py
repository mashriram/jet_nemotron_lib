# jet_nemotron/modeling/hybrid_model.py

import torch.nn as nn
from transformers import AutoModelForCausalLM

from jet_nemotron.blocks.jet_block import JetBlock


class JetNemotronHybridModel(nn.Module):
    """
    A hybrid model that programmatically replaces attention layers of a pre-trained
    Hugging Face model with JetBlock modules, following the PostNAS methodology.
    """

    def __init__(self, model_name: str, layers_to_replace: list[int]):
        super().__init__()

        print(f"Initializing hybrid model from base: {model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.config = self.base_model.config

        self._freeze_mlp_weights()
        self._replace_attention_layers(layers_to_replace)

    def _freeze_mlp_weights(self):
        """
        Freezes the MLP weights in all layers of the model, a core tenet of PostNAS
        to preserve learned knowledge and reduce training costs.
        """
        print("Freezing MLP weights...")
        for layer in self.base_model.model.layers:
            if hasattr(layer, "mlp"):
                for param in layer.mlp.parameters():
                    param.requires_grad = False

    def _replace_attention_layers(self, layers_to_replace: list[int]):
        """
        Replaces the standard self-attention mechanism with the JetBlock module
        for the specified layers.
        """
        print(f"Replacing attention layers at indices: {layers_to_replace}")
        for layer_idx in layers_to_replace:
            if 0 <= layer_idx < len(self.base_model.model.layers):
                # The new JetBlock uses the same model configuration.
                self.base_model.model.layers[layer_idx].self_attn = JetBlock(
                    self.config
                )
        print("Layer replacement complete.")

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def save_pretrained(self, save_directory: str):
        """Saves the fine-tuned model checkpoint."""
        self.base_model.save_pretrained(save_directory)
