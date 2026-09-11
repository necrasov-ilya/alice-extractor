"""Unit tests for checkpoint quantization primitives."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from alice_extractor.quantization import is_expert_tensor, quantize_int8_per_row, quantize_shard


class QuantizationTest(unittest.TestCase):
    def test_detects_only_expert_matrices(self):
        self.assertTrue(
            is_expert_tensor("model.encoder.layers.0.mlp.experts.mlp.w1")
        )
        self.assertTrue(
            is_expert_tensor("model.decoder.layers.4.mlp.experts.mlp.w2")
        )
        self.assertFalse(is_expert_tensor("model.encoder.layers.0.self_attn.q_proj.weight"))
        self.assertFalse(is_expert_tensor("model.encoder.layers.0.mlp.router.weight"))

    def test_quantizes_each_row_independently(self):
        values = torch.tensor(
            [[-1.0, -0.5, 0.0, 1.0], [-100.0, 10.0, 20.0, 50.0]],
            dtype=torch.bfloat16,
        )
        quantized, scale = quantize_int8_per_row(values)
        reconstructed = quantized.float() * scale[:, None]
        error = (reconstructed - values.float()).abs()
        self.assertEqual(quantized.dtype, torch.int8)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertTrue(torch.all(error <= scale[:, None] / 2 + 1e-6))

    def test_zero_row_remains_zero(self):
        quantized, scale = quantize_int8_per_row(torch.zeros(2, 4))
        self.assertTrue(torch.equal(quantized, torch.zeros_like(quantized)))
        self.assertTrue(torch.all(scale > 0))

    def test_quantizes_one_safetensors_shard_atomically(self):
        expert_name = "model.encoder.layers.0.mlp.experts.mlp.w1"
        dense_name = "model.encoder.embed_tokens.weight"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model-00001-of-00001.safetensors"
            destination = root / "quant_model-00001-of-00001.safetensors"
            save_file(
                {
                    expert_name: torch.tensor([[1.0, -0.5], [0.0, 2.0]]),
                    dense_name: torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
                },
                source,
            )

            expert_count, preserved_count = quantize_shard(source, destination)

            self.assertEqual((expert_count, preserved_count), (1, 1))
            self.assertTrue(destination.is_file())
            self.assertFalse((root / f".{destination.name}.partial").exists())
            with safe_open(destination, framework="pt") as checkpoint:
                self.assertEqual(checkpoint.get_tensor(expert_name).dtype, torch.int8)
                self.assertEqual(
                    checkpoint.get_tensor(f"{expert_name}_scale").dtype,
                    torch.float32,
                )
                self.assertEqual(checkpoint.get_tensor(dense_name).dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
