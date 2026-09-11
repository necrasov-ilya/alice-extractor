"""Inference-only MPS kernels for the mixed int8/BF16 AliceAI checkpoint.

The upstream grouped MoE fallback synchronizes with the CPU and launches one
matrix multiplication per expert. The first local prototype replaced that
fallback with a dense multiplication over all 512 experts, which is efficient
per operation but performs roughly 512 times more expert arithmetic than the
router selected.

This module selects only the routed expert matrices. Tokens are processed in
bounded chunks so temporary selected-weight tensors cannot grow with the whole
document length.
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class RoutedMoEOptions:
    """Memory and numerical options for routed expert execution."""

    token_chunk_size: int = 16

    def __post_init__(self) -> None:
        if self.token_chunk_size < 1:
            raise ValueError("token_chunk_size must be positive")


def routed_int8_swiglu(
    module,
    hidden_states: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    *,
    token_chunk_size: int,
) -> torch.Tensor:
    """Run only routed int8 experts and return token-major outputs.

    Expert weights are stored as per-output-row symmetric int8 matrices. MPS
    currently has no fused kernel for this custom layout, so each selected
    matrix is converted to the activation dtype immediately before its batched
    multiplication. At no point are all experts dequantized together.
    """

    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must be rank 2, got shape {tuple(hidden_states.shape)}")

    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return hidden_states.reshape(0, hidden_size)

    args = module.args
    num_experts = args.moe_num_experts
    intermediate_size = args.ffn_hidden_size
    top_k = expert_indices.shape[-1]

    if expert_indices.shape != (num_tokens, top_k):
        raise ValueError("expert_indices must contain one top-k row per token")
    if expert_weights.shape != expert_indices.shape:
        raise ValueError("expert_weights and expert_indices must have identical shapes")
    if top_k != args.moe_top_k:
        raise ValueError(f"expected top_k={args.moe_top_k}, got {top_k}")

    expected_shape = (num_experts * intermediate_size, hidden_size)
    for name in ("w1", "v1", "w2"):
        actual_shape = tuple(getattr(module, name).shape)
        if actual_shape != expected_shape:
            raise ValueError(f"{name} has shape {actual_shape}, expected {expected_shape}")

    dtype = hidden_states.dtype
    w1_all = module.w1.view(num_experts, intermediate_size, hidden_size)
    v1_all = module.v1.view(num_experts, intermediate_size, hidden_size)
    w2_all = module.w2.view(num_experts, intermediate_size, hidden_size)
    s1_all = module.w1_scale.view(num_experts, intermediate_size)
    sv_all = module.v1_scale.view(num_experts, intermediate_size)
    s2_all = module.w2_scale.view(num_experts, intermediate_size)

    chunk_outputs: list[torch.Tensor] = []
    for token_start in range(0, num_tokens, token_chunk_size):
        token_end = min(token_start + token_chunk_size, num_tokens)
        chunk_states = hidden_states[token_start:token_end]
        chunk_indices = expert_indices[token_start:token_end]
        chunk_weights = expert_weights[token_start:token_end]
        chunk_tokens = token_end - token_start

        expert_ids = chunk_indices.reshape(-1)
        selected_states = chunk_states.repeat_interleave(top_k, dim=0).unsqueeze(1)

        w1 = w1_all.index_select(0, expert_ids).to(dtype).transpose(1, 2).contiguous()
        s1 = s1_all.index_select(0, expert_ids).to(dtype)
        gate = torch.bmm(selected_states, w1).squeeze(1).mul_(s1)
        del w1, s1

        v1 = v1_all.index_select(0, expert_ids).to(dtype).transpose(1, 2).contiguous()
        sv = sv_all.index_select(0, expert_ids).to(dtype)
        up = torch.bmm(selected_states, v1).squeeze(1).mul_(sv)
        del v1, sv, selected_states

        activated = args.activation_fn(gate)
        activated.mul_(up)
        del gate, up
        activated.mul_(s2_all.index_select(0, expert_ids).to(dtype))

        w2 = w2_all.index_select(0, expert_ids).to(dtype)
        routed = torch.bmm(activated.unsqueeze(1), w2).squeeze(1)
        del activated, w2

        routed = routed.view(chunk_tokens, top_k, hidden_size)
        routing_weights = chunk_weights.to(dtype)
        reduced = routed.new_zeros((chunk_tokens, hidden_size))
        for top_k_position in range(top_k):
            reduced.add_(routed[:, top_k_position] * routing_weights[:, top_k_position, None])
        chunk_outputs.append(reduced)

    if len(chunk_outputs) == 1:
        return chunk_outputs[0]
    return torch.cat(chunk_outputs, dim=0)


def _routed_forward(self, hidden_states, expert_weights, expert_indices):
    options: RoutedMoEOptions = self._alice_routed_options
    return routed_int8_swiglu(
        self,
        hidden_states,
        expert_weights,
        expert_indices,
        token_chunk_size=options.token_chunk_size,
    )


def _dmoe_inference_forward(self, hidden_states):
    _, expert_weights, expert_indices = self.router(hidden_states)
    return self.experts(hidden_states, expert_weights, expert_indices)


def _mps_lm_head_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Avoid converting the full tied vocabulary matrix to FP32 per token."""

    flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    weight = self.out_proj.weight
    bias = self.out_proj.bias
    if flat_states.device.type != "mps":
        raise RuntimeError("The optimized language-model head is MPS-only")
    if flat_states.dtype != weight.dtype:
        flat_states = flat_states.to(weight.dtype)
    logits = F.linear(flat_states, weight, bias)
    # Generation code expects stable floating-point logits. Converting the
    # small result is much cheaper than converting the 135k x 1536 matrix.
    logits = logits.float()
    return logits.view(*hidden_states.shape[:-1], weight.shape[0])


def patch_model_for_mps(
    model,
    *,
    options: RoutedMoEOptions,
    optimize_lm_head: bool,
) -> tuple[int, int, int]:
    """Patch dynamically loaded AliceAI modules for inference.

    Returns counts of patched grouped expert blocks, dMoE wrappers, and
    language-model heads respectively.
    """

    grouped_count = 0
    dmoe_count = 0
    lm_head_count = 0

    for module in model.modules():
        class_name = type(module).__name__
        if class_name == "GroupedSwiGLU":
            module._alice_routed_options = options
            module.forward = types.MethodType(_routed_forward, module)
            grouped_count += 1
        elif class_name == "dMoE":
            module.forward = types.MethodType(_dmoe_inference_forward, module)
            dmoe_count += 1
        elif optimize_lm_head and class_name == "AliceAIT5LMHead":
            module.forward = types.MethodType(_mps_lm_head_forward, module)
            lm_head_count += 1

    if grouped_count == 0 or dmoe_count == 0:
        raise RuntimeError("AliceAI MoE modules were not found in the loaded model")
    if optimize_lm_head and lm_head_count != 1:
        raise RuntimeError(f"expected one AliceAIT5LMHead, patched {lm_head_count}")
    return grouped_count, dmoe_count, lm_head_count


__all__ = ["RoutedMoEOptions", "patch_model_for_mps", "routed_int8_swiglu"]
