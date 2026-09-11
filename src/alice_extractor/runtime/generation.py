"""Prompt preparation and deterministic AliceAI generation."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .model import LoadedModel


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    input_tokens: int
    generated_tokens: int
    elapsed_seconds: float

    @property
    def end_to_end_tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.generated_tokens / self.elapsed_seconds


class AliceGenerator:
    """Deterministic text-to-text generator using AliceAI sentinel tokens."""

    def __init__(self, loaded: LoadedModel):
        self.loaded = loaded
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer
        self.device = loaded.options.device
        self.mode_id = self.tokenizer.convert_tokens_to_ids("[_S_]")
        self.span_id = self.tokenizer.convert_tokens_to_ids("<SPAN#0>")
        self.bos_id = self.model.config.decoder.bos_token_id

    @staticmethod
    def question_prompt(text: str) -> str:
        return text if "Ответ:" in text else f"Вопрос: {text}\nОтвет: "

    def generate(self, prompt: str, *, max_new_tokens: int = 96) -> GenerationResult:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        content_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        input_ids = torch.tensor(
            [[self.mode_id, *content_ids, self.span_id]],
            dtype=torch.long,
            device=self.device,
        )
        decoder_prefix = torch.tensor(
            [[self.bos_id, self.span_id]],
            dtype=torch.long,
            device=self.device,
        )

        torch.mps.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                decoder_input_ids=decoder_prefix,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        torch.mps.synchronize()
        elapsed = time.perf_counter() - started

        completion_ids = output_ids[0, decoder_prefix.shape[1] :]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        text = text.split("<SPAN#", 1)[0]
        return GenerationResult(
            text=text,
            input_tokens=len(content_ids),
            generated_tokens=len(completion_ids),
            elapsed_seconds=elapsed,
        )


__all__ = ["AliceGenerator", "GenerationResult"]

