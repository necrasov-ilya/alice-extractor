"""Reproducible phase-by-phase benchmark for the local MPS runtime."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .runtime import RuntimeOptions, RuntimePaths, load_model


DEFAULT_PROMPT = (
    "Документ: Соглашение об оказании услуг. Исполнитель предоставляет доступ "
    "к системе Алиса в течение 5 рабочих дней. Стоимость доступа составляет "
    "1 250 000 рублей. Срок действия соглашения составляет 12 месяцев.\n"
    "Вопрос: Какова стоимость доступа?\n"
    "Ответ: "
)


@dataclass(frozen=True, slots=True)
class PhaseResult:
    input_tokens: int
    generated_tokens: int
    encoder_seconds: float
    decoder_prefill_seconds: float
    cached_decode_seconds: float
    cached_decode_tokens_per_second: float
    end_to_end_seconds: float
    end_to_end_tokens_per_second: float
    output_preview: str


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    load_seconds: float
    token_chunk_size: int
    optimized_lm_head: bool
    cold_start: PhaseResult
    warmed: PhaseResult


def _synchronized_seconds(operation):
    torch.mps.synchronize()
    started = time.perf_counter()
    result = operation()
    torch.mps.synchronize()
    return result, time.perf_counter() - started


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark AliceAI MPS runtime phases")
    parser.add_argument("--model-dir", type=Path, default=Path("local/aliceai-t5/model"))
    parser.add_argument("--weights-dir", type=Path, default=Path("local/aliceai-t5/data/int8"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--token-chunk-size", type=int, default=16)
    parser.add_argument("--reference-lm-head", action="store_true")
    parser.add_argument("--output", type=Path, help="optionally write the JSON result to this file")
    args = parser.parse_args(argv)
    if args.decode_steps < 1:
        parser.error("--decode-steps must be positive")
    return args


def _run_phases(model, tokenizer, prompt: str, decode_steps: int) -> PhaseResult:
    content_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    mode_id = tokenizer.convert_tokens_to_ids("[_S_]")
    span_id = tokenizer.convert_tokens_to_ids("<SPAN#0>")
    bos_id = model.config.decoder.bos_token_id
    input_ids = torch.tensor([[mode_id, *content_ids, span_id]], dtype=torch.long, device="mps")
    attention_mask = torch.ones_like(input_ids)
    decoder_prefix = torch.tensor([[bos_id, span_id]], dtype=torch.long, device="mps")

    generated_ids: list[int] = []
    with torch.inference_mode():
        encoder_outputs, encoder_seconds = _synchronized_seconds(
            lambda: model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        )

        first_output, decoder_prefill_seconds = _synchronized_seconds(
            lambda: model(
                attention_mask=attention_mask,
                decoder_input_ids=decoder_prefix,
                encoder_outputs=encoder_outputs,
                use_cache=True,
                logits_to_keep=1,
            )
        )
        next_token = first_output.logits[:, -1:].argmax(dim=-1)
        generated_ids.append(int(next_token.item()))
        past_key_values = first_output.past_key_values

        def cached_decode():
            nonlocal next_token, past_key_values
            for _ in range(decode_steps):
                output = model(
                    attention_mask=attention_mask,
                    decoder_input_ids=next_token,
                    encoder_outputs=encoder_outputs,
                    past_key_values=past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
                next_token = output.logits[:, -1:].argmax(dim=-1)
                past_key_values = output.past_key_values
                generated_ids.append(int(next_token.item()))

        _, cached_decode_seconds = _synchronized_seconds(cached_decode)

    generated_tokens = decode_steps + 1
    measured_total = encoder_seconds + decoder_prefill_seconds + cached_decode_seconds
    preview = tokenizer.decode(generated_ids, skip_special_tokens=True).split("<SPAN#", 1)[0]
    return PhaseResult(
        input_tokens=input_ids.shape[1],
        generated_tokens=generated_tokens,
        encoder_seconds=encoder_seconds,
        decoder_prefill_seconds=decoder_prefill_seconds,
        cached_decode_seconds=cached_decode_seconds,
        cached_decode_tokens_per_second=decode_steps / cached_decode_seconds,
        end_to_end_seconds=measured_total,
        end_to_end_tokens_per_second=generated_tokens / measured_total,
        output_preview=preview,
    )


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    options = RuntimeOptions(
        token_chunk_size=args.token_chunk_size,
        optimize_lm_head=not args.reference_lm_head,
    )
    load_started = time.perf_counter()
    loaded = load_model(RuntimePaths(args.model_dir, args.weights_dir), options)
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - load_started

    cold_start = _run_phases(loaded.model, loaded.tokenizer, args.prompt, args.decode_steps)
    warmed = _run_phases(loaded.model, loaded.tokenizer, args.prompt, args.decode_steps)
    return BenchmarkResult(
        load_seconds=load_seconds,
        token_chunk_size=args.token_chunk_size,
        optimized_lm_head=not args.reference_lm_head,
        cold_start=cold_start,
        warmed=warmed,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_benchmark(args)
    payload = json.dumps(asdict(result), ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
