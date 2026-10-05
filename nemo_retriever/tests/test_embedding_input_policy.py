# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_retriever.models.inference.embedding_input import EmbeddingInputPolicy


class _LossyDecodeTokenizer:
    """Tokenizer whose decoded token slices normalize source whitespace."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        tokens = [ord(character) for character in text]
        return ([1] + tokens + [2]) if add_special_tokens else tokens

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return "".join(chr(token_id) for token_id in token_ids).replace("[\u200c", "[").replace("<\u200c", "<")


def test_split_falls_back_to_exact_source_slices_when_decode_normalizes_text() -> None:
    policy = EmbeddingInputPolicy(
        tokenizer=_LossyDecodeTokenizer(),
        max_tokens=14,
        prefix="passage:",
    )
    text = "abc[\u200cdefghijklmnop"

    plan = policy.plan([text])[0]

    assert plan.requires_split
    assert "".join(child.content for child in plan.children) == text
    assert all(policy._formatted_token_count(child.content) <= policy.max_tokens for child in plan.children)


def test_exact_source_fallback_preserves_token_ranges() -> None:
    policy = EmbeddingInputPolicy(
        tokenizer=_LossyDecodeTokenizer(),
        max_tokens=7,
        prefix="",
    )
    text = "ab<\u200ccdefghijkl"

    children = policy._split(text)

    assert children[0].start_token == 0
    assert all(left.end_token == right.start_token for left, right in zip(children, children[1:]))
    assert children[-1].end_token == sum(
        len(policy.tokenizer.encode(child.content, add_special_tokens=False)) for child in children
    )
