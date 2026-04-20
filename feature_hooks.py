import torch
import torch.nn as nn

class FeatureExtractor:
    """
    Utility to hook intermediate features from a model without modification.
    """
    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names
        self.features = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                self.hooks.append(module.register_forward_hook(self._get_hook(name)))

    def _get_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def get_features(self):
        return [self.features[name] for name in self.layer_names]

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
