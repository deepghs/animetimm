import logging
from typing import Literal

import torch
from torch import nn


class ModuleWrapper(nn.Module):
    def __init__(self, base_module: nn.Module, classifier: nn.Module,
                 wrap_mode: Literal['softmax', 'sigmoid'] = 'softmax'):
        super().__init__()
        self.base_module = base_module
        self.classifier = classifier

        self._output_features = None
        if wrap_mode not in {'sigmoid', 'softmax'}:
            raise ValueError(f'Unknown wrap mode - {wrap_mode!r}.')
        self._wrap_mode = wrap_mode
        self._register_hook()

    def _register_hook(self):
        def hook_fn(module, input_tensor, output_tensor):
            assert isinstance(input_tensor, tuple) and len(input_tensor) == 1
            input_tensor = input_tensor[0]
            self._output_features = input_tensor

        self.classifier.register_forward_hook(hook_fn)

    def forward(self, x: torch.Tensor):
        logits = self.base_module(x)
        if self._wrap_mode == 'softmax':
            preds = torch.softmax(logits, dim=-1)
        elif self._wrap_mode == 'sigmoid':
            preds = torch.sigmoid(logits)
        else:
            assert False, 'Should not reach this line'

        if self._output_features is None:
            raise RuntimeError("Target module did not receive any input during forward pass")
        features, self._output_features = self._output_features, None
        assert all([x == 1 for x in features.shape[2:]]), f'Invalid feature shape: {features.shape!r}'
        features = torch.flatten(features, start_dim=1)

        return features, logits, preds


def wrap_timm_model(model, wrap_mode: Literal['softmax', 'sigmoid'] = 'softmax'):
    classifier = model.get_classifier()
    classifier_position = None
    for name, module in model.named_modules():
        if module is classifier:
            classifier_position = name
            break
    if not classifier_position:
        raise RuntimeError(f'No classifier module found in model {type(model)}.')
    logging.info(f'Classifier module found at {classifier_position!r}:\n{classifier}')

    wrapped_model = ModuleWrapper(model, classifier=classifier, wrap_mode=wrap_mode)
    return wrapped_model
