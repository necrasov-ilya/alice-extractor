"""Construction and loading of the local mixed int8/BF16 AliceAI model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from .mps_moe import RoutedMoEOptions, patch_model_for_mps


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    model_dir: Path
    weights_dir: Path

    def normalized(self) -> "RuntimePaths":
        return RuntimePaths(self.model_dir.expanduser().resolve(), self.weights_dir.expanduser().resolve())

    def validate(self) -> list[Path]:
        paths = self.normalized()
        required = [
            paths.model_dir / "config.json",
            paths.model_dir / "tokenizer.json",
            paths.model_dir / "modeling_aliceai_t5.py",
            paths.model_dir / "modeling_aliceai_t5_moe.py",
            paths.model_dir / "moe_layers.py",
        ]
        missing = [path for path in required if not path.is_file()]
        shards = sorted(paths.weights_dir.glob("quant_model-*.safetensors"))
        if len(shards) != 15:
            missing.append(paths.weights_dir / f"<expected 15 quantized shards, found {len(shards)}>")
        return missing


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    token_chunk_size: int = 16
    optimize_lm_head: bool = True
    device: str = "mps"

    def __post_init__(self) -> None:
        if self.device != "mps":
            raise ValueError("The mixed int8 runtime currently supports only MPS")
        if self.token_chunk_size < 1:
            raise ValueError("token_chunk_size must be positive")


@dataclass(slots=True)
class LoadedModel:
    model: object
    tokenizer: object
    paths: RuntimePaths
    options: RuntimeOptions
    patched_grouped_blocks: int
    patched_dmoe_blocks: int
    patched_lm_heads: int


def _replace_expert_parameters_with_int8(model) -> int:
    grouped_blocks = 0
    for module in model.modules():
        if type(module).__name__ != "GroupedSwiGLU":
            continue
        grouped_blocks += 1
        for name in ("w1", "v1", "w2"):
            parameter = getattr(module, name)
            setattr(
                module,
                name,
                torch.nn.Parameter(torch.empty(parameter.shape, dtype=torch.int8), requires_grad=False),
            )
            module.register_buffer(
                f"{name}_scale",
                torch.empty(parameter.shape[0], dtype=torch.float32),
                persistent=False,
            )
    if grouped_blocks == 0:
        raise RuntimeError("no GroupedSwiGLU blocks found while preparing int8 parameters")
    return grouped_blocks


def _load_weights(model, weights_dir: Path) -> int:
    parameters = dict(model.named_parameters())
    buffers: dict[str, torch.Tensor] = {}
    for prefix, module in model.named_modules():
        for name, tensor in module._buffers.items():
            if tensor is not None:
                full_name = f"{prefix}.{name}" if prefix else name
                buffers[full_name] = tensor

    seen_parameters: set[str] = set()
    shards = sorted(weights_dir.glob("quant_model-*.safetensors"))
    if len(shards) != 15:
        raise FileNotFoundError(f"expected 15 quantized shards in {weights_dir}, found {len(shards)}")

    with torch.inference_mode():
        for shard in shards:
            with safe_open(shard, framework="pt") as source:
                for name in source.keys():
                    value = source.get_tensor(name)
                    if name.endswith("_scale"):
                        target = buffers.get(name)
                        if target is None:
                            raise KeyError(f"checkpoint buffer has no model target: {name}")
                        target.copy_(value)
                        seen_parameters.add(name.removesuffix("_scale"))
                        continue

                    target = parameters.get(name)
                    if target is None:
                        target = buffers.get(name)
                    if target is None:
                        raise KeyError(f"checkpoint tensor has no model target: {name}")

                    if value.dtype == torch.int8:
                        target.copy_(value)
                    else:
                        target.copy_(value.to(torch.bfloat16))
                    seen_parameters.add(name)

    missing = sorted(name for name in parameters if name not in seen_parameters)
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"{len(missing)} model parameters were not loaded: {preview}")
    return len(shards)


def load_model(paths: RuntimePaths, options: RuntimeOptions | None = None) -> LoadedModel:
    """Load a local AliceAI checkpoint directly onto Apple Silicon MPS."""

    options = options or RuntimeOptions()
    paths = paths.normalized()
    missing = paths.validate()
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"local AliceAI assets are incomplete:\n{formatted}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available. Run this command on a supported Apple Silicon macOS host")

    config = AutoConfig.from_pretrained(paths.model_dir, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=True)

    prepared_blocks = _replace_expert_parameters_with_int8(model)
    model.to_empty(device=options.device)

    patched_grouped, patched_dmoe, patched_heads = patch_model_for_mps(
        model,
        options=RoutedMoEOptions(token_chunk_size=options.token_chunk_size),
        optimize_lm_head=options.optimize_lm_head,
    )
    if prepared_blocks != patched_grouped:
        raise RuntimeError(
            f"prepared {prepared_blocks} int8 blocks but patched {patched_grouped} grouped blocks"
        )

    _load_weights(model, paths.weights_dir)
    torch.mps.synchronize()
    torch.mps.empty_cache()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(paths.model_dir, trust_remote_code=True)

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        paths=paths,
        options=options,
        patched_grouped_blocks=patched_grouped,
        patched_dmoe_blocks=patched_dmoe,
        patched_lm_heads=patched_heads,
    )


__all__ = ["LoadedModel", "RuntimeOptions", "RuntimePaths", "load_model"]
