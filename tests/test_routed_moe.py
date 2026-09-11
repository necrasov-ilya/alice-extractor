"""Numerical tests for routed expert execution without MPS dependencies."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import torch

from alice_extractor.runtime.mps_moe import routed_int8_swiglu


@dataclass
class _Args:
    moe_num_experts: int = 5
    ffn_hidden_size: int = 4
    moe_top_k: int = 2
    activation_fn = staticmethod(torch.nn.functional.silu)


class _Module:
    def __init__(self):
        self.args = _Args()
        rows = self.args.moe_num_experts * self.args.ffn_hidden_size
        hidden = 6
        generator = torch.Generator().manual_seed(7)
        self.w1 = torch.randint(-10, 11, (rows, hidden), dtype=torch.int8, generator=generator)
        self.v1 = torch.randint(-10, 11, (rows, hidden), dtype=torch.int8, generator=generator)
        self.w2 = torch.randint(-10, 11, (rows, hidden), dtype=torch.int8, generator=generator)
        self.w1_scale = torch.rand(rows, generator=generator) * 0.04
        self.v1_scale = torch.rand(rows, generator=generator) * 0.04
        self.w2_scale = torch.rand(rows, generator=generator) * 0.04


def _reference(module, hidden_states, expert_weights, expert_indices):
    args = module.args
    num_tokens, hidden = hidden_states.shape
    intermediate = args.ffn_hidden_size
    output = hidden_states.new_zeros((num_tokens, hidden))
    for token in range(num_tokens):
        for position in range(args.moe_top_k):
            expert = int(expert_indices[token, position])
            start = expert * intermediate
            end = start + intermediate
            w1 = module.w1[start:end].float() * module.w1_scale[start:end, None]
            v1 = module.v1[start:end].float() * module.v1_scale[start:end, None]
            w2 = module.w2[start:end].float() * module.w2_scale[start:end, None]
            gate = torch.nn.functional.silu(hidden_states[token] @ w1.t())
            up = hidden_states[token] @ v1.t()
            expert_output = (gate * up) @ w2
            output[token].add_(expert_output * expert_weights[token, position])
    return output


class RoutedMoETest(unittest.TestCase):
    def setUp(self):
        self.module = _Module()
        generator = torch.Generator().manual_seed(11)
        self.hidden_states = torch.randn(7, 6, generator=generator)
        self.expert_indices = torch.tensor(
            [[0, 3], [1, 4], [2, 0], [4, 3], [1, 2], [0, 4], [3, 1]],
            dtype=torch.long,
        )
        raw_weights = torch.rand(7, 2, generator=generator)
        self.expert_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)

    def test_matches_per_expert_reference(self):
        expected = _reference(
            self.module,
            self.hidden_states,
            self.expert_weights,
            self.expert_indices,
        )
        actual = routed_int8_swiglu(
            self.module,
            self.hidden_states,
            self.expert_weights,
            self.expert_indices,
            token_chunk_size=3,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_chunk_size_does_not_change_output(self):
        outputs = [
            routed_int8_swiglu(
                self.module,
                self.hidden_states,
                self.expert_weights,
                self.expert_indices,
                token_chunk_size=chunk_size,
            )
            for chunk_size in (1, 2, 7, 20)
        ]
        for output in outputs[1:]:
            torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)

    def test_supports_bfloat16_activations(self):
        hidden_states = self.hidden_states.to(torch.bfloat16)
        actual = routed_int8_swiglu(
            self.module,
            hidden_states,
            self.expert_weights,
            self.expert_indices,
            token_chunk_size=3,
        )
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(actual.shape, hidden_states.shape)
        self.assertTrue(torch.isfinite(actual).all())

    def test_rejects_invalid_chunk_size(self):
        from alice_extractor.runtime.mps_moe import RoutedMoEOptions

        with self.assertRaises(ValueError):
            RoutedMoEOptions(token_chunk_size=0)


if __name__ == "__main__":
    unittest.main()
