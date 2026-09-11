"""Streaming mixed int8/BF16 checkpoint preparation for AliceAI-T5."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_REPO_ID = "yandex/AliceAI-T5-35B-A0.6B"
CHECKPOINT_FORMAT = "alice-extractor-mixed-int8-bf16-v1"
RUNTIME_URL = "https://github.com/necrasov-ilya/alice-extractor"
_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
METADATA_PATTERNS = (
    ".gitattributes",
    "CONTRIBUTING.md",
    "LICENSE",
    "NOTICES",
    "README.md",
    "config.json",
    "configuration_*.py",
    "finetune_example.py",
    "generation_config.json",
    "model.safetensors.index.json",
    "modeling_*.py",
    "moe_layers.py",
    "requirements*.txt",
    "tokenizer.json",
    "tokenizer_config.json",
)


@dataclass(frozen=True, slots=True)
class QuantizationOptions:
    repo_id: str
    revision: str
    model_dir: Path
    source_dir: Path
    output_dir: Path
    keep_source_shards: bool = False
    force: bool = False


def is_expert_tensor(name: str) -> bool:
    return name.endswith((".w1", ".v1", ".w2")) and ".mlp.experts.mlp." in name


def quantize_int8_per_row(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically quantize the final dimension with one scale per row."""

    values = tensor.float()
    scale = (values.abs().amax(dim=-1) / 127.0).clamp_min(1e-8)
    quantized = (values / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


def _download_metadata(options: QuantizationOptions) -> str:
    options.model_dir.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(options.repo_id, revision=options.revision)
    snapshot_download(
        repo_id=options.repo_id,
        revision=info.sha,
        local_dir=options.model_dir,
        allow_patterns=list(METADATA_PATTERNS),
    )
    return info.sha


def _checkpoint_shards(model_dir: Path) -> list[str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"checkpoint index is missing: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"checkpoint index has no weight_map: {index_path}")
    return sorted(set(weight_map.values()))


def quantize_shard(source: Path, destination: Path) -> tuple[int, int]:
    """Quantize expert tensors in one Safetensors shard and write atomically."""

    tensors: dict[str, torch.Tensor] = {}
    expert_count = 0
    preserved_count = 0
    with safe_open(source, framework="pt") as checkpoint:
        for name in checkpoint.keys():
            tensor = checkpoint.get_tensor(name)
            if is_expert_tensor(name):
                quantized, scale = quantize_int8_per_row(tensor)
                tensors[name] = quantized
                tensors[f"{name}_scale"] = scale
                expert_count += 1
            else:
                tensors[name] = tensor
                preserved_count += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    save_file(tensors, temporary)
    temporary.replace(destination)
    return expert_count, preserved_count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes(shape: list[int], dtype: str) -> int:
    try:
        item_size = _DTYPE_BYTES[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported Safetensors dtype in release metadata: {dtype}") from error
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * item_size


def write_release_metadata(
    *,
    model_dir: Path,
    output_dir: Path,
    base_model: str,
    revision: str,
) -> dict[str, object]:
    """Write a loadable shard index, checksums, and a public format manifest."""

    source_index_path = model_dir / "model.safetensors.index.json"
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    source_metadata = source_index.get("metadata", {})
    shards = sorted(output_dir.glob("quant_model-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no quantized shards found in {output_dir}")

    weight_map: dict[str, str] = {}
    total_size = 0
    shard_records: list[dict[str, object]] = []
    checksum_lines: list[str] = []
    for shard in shards:
        tensor_count = 0
        scale_tensor_count = 0
        with safe_open(shard, framework="pt") as checkpoint:
            for name in checkpoint.keys():
                if name in weight_map:
                    raise ValueError(f"tensor appears in more than one shard: {name}")
                tensor = checkpoint.get_slice(name)
                weight_map[name] = shard.name
                total_size += _tensor_bytes(tensor.get_shape(), tensor.get_dtype())
                tensor_count += 1
                scale_tensor_count += int(name.endswith("_scale"))

        checksum = _sha256(shard)
        checksum_lines.append(f"{checksum}  {shard.name}")
        shard_records.append(
            {
                "file": shard.name,
                "size_bytes": shard.stat().st_size,
                "sha256": checksum,
                "tensor_count": tensor_count,
                "scale_tensor_count": scale_tensor_count,
            }
        )

    checkpoint_index = {
        "metadata": {
            "format": CHECKPOINT_FORMAT,
            "total_parameters": source_metadata.get("total_parameters"),
            "total_size": total_size,
        },
        "weight_map": weight_map,
    }
    (output_dir / "quant_model.safetensors.index.json").write_text(
        json.dumps(checkpoint_index, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    manifest = {
        "format": CHECKPOINT_FORMAT,
        "base_model": base_model,
        "resolved_revision": revision,
        "runtime": RUNTIME_URL,
        "expert_tensors": "w1, v1 and w2 under .mlp.experts.mlp.",
        "expert_quantization": "symmetric int8 per output row",
        "scale_dtype": "float32",
        "non_expert_dtype": "preserved from the source checkpoint",
        "shard_count": len(shards),
        "tensor_count": len(weight_map),
        "checkpoint_size_bytes": total_size,
        "shards": shard_records,
    }
    (output_dir / "quantization_config.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def prepare_checkpoint(options: QuantizationOptions) -> dict[str, object]:
    options.source_dir.mkdir(parents=True, exist_ok=True)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_revision = _download_metadata(options)
    shards = _checkpoint_shards(options.model_dir)
    started = time.perf_counter()
    completed: list[dict[str, object]] = []

    for position, shard_name in enumerate(shards, start=1):
        output_name = f"quant_{shard_name}"
        destination = options.output_dir / output_name
        if destination.is_file() and not options.force:
            print(f"[{position}/{len(shards)}] already present: {output_name}", flush=True)
            completed.append({"file": output_name, "status": "existing"})
            continue

        shard_started = time.perf_counter()
        source = Path(
            hf_hub_download(
                repo_id=options.repo_id,
                filename=shard_name,
                revision=resolved_revision,
                local_dir=options.source_dir,
            )
        )
        expert_count, preserved_count = quantize_shard(source, destination)
        if not options.keep_source_shards:
            source.unlink(missing_ok=True)
        gc.collect()
        elapsed = time.perf_counter() - shard_started
        print(
            f"[{position}/{len(shards)}] {output_name}: "
            f"experts={expert_count} preserved={preserved_count} time={elapsed:.1f}s",
            flush=True,
        )
        completed.append(
            {
                "file": output_name,
                "status": "quantized",
                "expert_tensors": expert_count,
                "preserved_tensors": preserved_count,
                "seconds": elapsed,
            }
        )

    manifest = write_release_metadata(
        model_dir=options.model_dir,
        output_dir=options.output_dir,
        base_model=options.repo_id,
        revision=resolved_revision,
    )
    manifest["requested_revision"] = options.revision
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["preparation"] = completed
    (options.output_dir / "quantization_config.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and quantize AliceAI-T5 one source shard at a time"
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main", help="prefer a commit hash for a release build")
    parser.add_argument("--model-dir", type=Path, default=Path("local/aliceai-t5/model"))
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("local/aliceai-t5/data/source"),
        help="temporary directory for one source shard at a time",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("local/aliceai-t5/data/int8"))
    parser.add_argument("--keep-source-shards", action="store_true")
    parser.add_argument("--force", action="store_true", help="replace existing quantized shards")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    options = QuantizationOptions(
        repo_id=args.repo_id,
        revision=args.revision,
        model_dir=args.model_dir.expanduser().resolve(),
        source_dir=args.source_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        keep_source_shards=args.keep_source_shards,
        force=args.force,
    )
    manifest = prepare_checkpoint(options)
    print(
        f"completed {manifest['shard_count']} shards in "
        f"{manifest['elapsed_seconds']:.1f}s at {options.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "QuantizationOptions",
    "is_expert_tensor",
    "main",
    "prepare_checkpoint",
    "quantize_int8_per_row",
    "quantize_shard",
    "write_release_metadata",
]
