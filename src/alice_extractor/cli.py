"""Command-line entry point for the local AliceAI runtime."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .runtime import AliceGenerator, RuntimeOptions, RuntimePaths, load_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AliceAI-T5 locally on Apple Silicon")
    parser.add_argument(
        "prompt",
        nargs="?",
        help="question or complete AliceAI prompt",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("local/aliceai-t5/model"),
        help="directory with config, tokenizer, and custom model code",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("local/aliceai-t5/data/int8"),
        help="directory with 15 quant_model safetensors shards",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--token-chunk-size", type=int, default=16)
    parser.add_argument("--raw-prompt", action="store_true", help="do not add the question/answer template")
    parser.add_argument("--repl", action="store_true", help="start an interactive prompt loop")
    parser.add_argument(
        "--reference-lm-head",
        action="store_true",
        help="use the original FP32 language-model head for numerical comparison",
    )
    return parser


def _print_result(generator: AliceGenerator, prompt: str, args: argparse.Namespace) -> None:
    prepared = prompt if args.raw_prompt else generator.question_prompt(prompt)
    result = generator.generate(prepared, max_new_tokens=args.max_new_tokens)
    print(result.text)
    print(
        f"input={result.input_tokens} generated={result.generated_tokens} "
        f"total={result.elapsed_seconds:.2f}s "
        f"end_to_end={result.end_to_end_tokens_per_second:.2f} tok/s",
        file=sys.stderr,
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = RuntimePaths(args.model_dir, args.weights_dir)
    options = RuntimeOptions(
        token_chunk_size=args.token_chunk_size,
        optimize_lm_head=not args.reference_lm_head,
    )

    started = time.perf_counter()
    loaded = load_model(paths, options)
    load_seconds = time.perf_counter() - started
    print(
        f"loaded on mps in {load_seconds:.1f}s, "
        f"patched_moe={loaded.patched_grouped_blocks}, "
        f"fast_lm_head={bool(loaded.patched_lm_heads)}",
        file=sys.stderr,
        flush=True,
    )
    generator = AliceGenerator(loaded)

    if args.prompt is not None:
        _print_result(generator, args.prompt, args)
        return 0
    if not args.repl and not sys.stdin.isatty():
        _print_result(generator, "Кто написал роман Война и мир?", args)
        return 0

    print("Модель готова. q или Ctrl-D завершает работу.", flush=True)
    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt or prompt.lower() in {"q", "quit", "exit"}:
            return 0
        _print_result(generator, prompt, args)


if __name__ == "__main__":
    raise SystemExit(main())
