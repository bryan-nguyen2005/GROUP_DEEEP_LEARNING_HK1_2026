"""Forward and visualization contract checks for the six architectures."""

import unittest

import torch
from torch import nn

from src.models import MODEL_NAMES, build_model


class ModelTests(unittest.TestCase):
    def test_forward_and_conv_hooks(self):
        inputs = torch.randn(4, 1, 48, 48, generator=torch.Generator().manual_seed(42))
        print(f"\n{'Model':<18} {'Trainable parameters':>20}")
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(name, pretrained=False).eval()
                first = model.get_first_conv_layer()
                final = model.get_final_conv_layer()
                self.assertIsInstance(first, nn.Conv2d)
                self.assertIsInstance(final, nn.Conv2d)
                observed = []
                hooks = [
                    layer.register_forward_hook(
                        lambda module, args, output: observed.append(output.shape)
                    )
                    for layer in (first, final)
                ]
                try:
                    with torch.inference_mode():
                        logits = model(inputs)
                finally:
                    for hook in hooks:
                        hook.remove()
                self.assertEqual(logits.shape, (4, 7))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertEqual(len(observed), 2)
                self.assertEqual(observed[0][-2:], (48, 48))
                self.assertGreater(observed[1][-2], 1)
                params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"{name:<18} {params:>20,}")
                del model, first, final

    def test_no_bn_and_activation_options(self):
        model = build_model("ImprovedFERNet", activation="leakyrelu", use_bn=False)
        self.assertFalse(any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules()))
        activations = [layer for layer in model.modules() if isinstance(layer, nn.LeakyReLU)]
        self.assertTrue(activations)
        self.assertTrue(all(layer.negative_slope == 0.1 for layer in activations))
