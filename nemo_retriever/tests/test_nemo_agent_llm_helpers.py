# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers used by the LLM backends.

Redaction and key resolution are security-adjacent: a regression here leaks a
credential into a log or an agent trajectory rather than failing a test loudly.
"""

from __future__ import annotations

import pytest

from nemo_retriever._agentic.nemo_agent.llm.helpers import (
    BODY_EXCERPT_CHARS,
    excerpt,
    redact_url,
    redact_urls_in_text,
    resolve_api_key,
)


class TestRedactUrl:
    def test_drops_userinfo_query_and_fragment(self):
        out = redact_url("https://user:pw@host:8443/v1/chat/completions?token=abc#frag")
        assert out == "https://host:8443/v1/chat/completions"

    def test_keeps_host_and_path(self):
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        assert redact_url(url) == url

    def test_each_url_in_a_comma_separated_list_is_redacted(self):
        # The endpoint option accepts a comma-separated list. A bare urlsplit would
        # leave every URL after the first sitting in `.path` with its userinfo
        # intact, so each segment has to be redacted independently.
        out = redact_url("https://a:pw@h1/v1/chat/completions, https://b:pw2@h2/v1/chat/completions")
        assert out == "https://h1/v1/chat/completions,https://h2/v1/chat/completions"
        assert "pw" not in out

    @pytest.mark.parametrize("value", ["", None])
    def test_empty_input_does_not_raise(self, value):
        assert redact_url(value) == ""

    def test_unparseable_input_degrades_instead_of_raising(self):
        # Redaction runs on error paths; raising here would mask the real failure.
        assert isinstance(redact_url("http://[unclosed"), str)


class TestRedactUrlsInText:
    """Scrubbing URLs out of free text we did not format ourselves."""

    def test_full_url_in_prose_loses_userinfo_and_query(self):
        out = redact_urls_in_text("failed calling https://user:pw@h.invalid/v1/chat?api_key=SECRET after 3 tries")
        assert "SECRET" not in out
        assert "user:pw" not in out
        assert "https://h.invalid/v1/chat" in out
        assert out.endswith("after 3 tries")

    def test_bare_path_query_is_scrubbed(self):
        # urllib3 connection errors carry no scheme or host, so the URL pattern
        # never matches them — yet the query string, where a credential actually
        # lives, is right there. This is the case that motivated the helper.
        text = "HTTPSConnectionPool(host='h', port=443): Max retries exceeded with url: /v1/chat?api_key=SECRET"
        out = redact_urls_in_text(text)
        assert "SECRET" not in out
        assert "?<redacted>" in out

    def test_trailing_sentence_punctuation_is_not_swallowed(self):
        out = redact_urls_in_text("could not reach https://h.invalid/v1?k=SECRET. Retrying.")
        assert "SECRET" not in out
        assert out.endswith(". Retrying.")

    def test_prose_question_mark_is_left_alone(self):
        # The `=` requirement is what keeps ordinary prose intact.
        text = "Did the endpoint respond? No route to host."
        assert redact_urls_in_text(text) == text

    def test_multiple_urls_all_scrubbed(self):
        out = redact_urls_in_text("tried https://a.invalid/v1?k=S1 then https://b.invalid/v1?k=S2")
        assert "S1" not in out and "S2" not in out

    def test_text_without_urls_is_unchanged(self):
        text = "maximum context length is 8192 tokens"
        assert redact_urls_in_text(text) == text

    @pytest.mark.parametrize("value", ["", None])
    def test_empty_input(self, value):
        assert redact_urls_in_text(value) == ""


class TestExcerpt:
    def test_short_text_passes_through_unchanged(self):
        assert excerpt("boom") == "boom"

    def test_long_text_is_truncated_with_an_ellipsis(self):
        out = excerpt("x" * (BODY_EXCERPT_CHARS + 500))
        assert len(out) == BODY_EXCERPT_CHARS + 3
        assert out.endswith("...")

    def test_exactly_at_the_limit_is_not_truncated(self):
        out = excerpt("x" * BODY_EXCERPT_CHARS)
        assert out == "x" * BODY_EXCERPT_CHARS

    @pytest.mark.parametrize("value", [None, ""])
    def test_empty_becomes_an_empty_string(self, value):
        assert excerpt(value) == ""

    def test_non_string_is_coerced(self):
        assert excerpt(404) == "404"


class TestResolveApiKey:
    def test_literal_key_passes_through(self):
        assert resolve_api_key("sk-literal") == "sk-literal"

    def test_env_indirection_is_followed_and_stripped(self, monkeypatch):
        monkeypatch.setenv("HELPER_TEST_KEY", "  sk-from-env\n")
        assert resolve_api_key("os.environ/HELPER_TEST_KEY") == "sk-from-env"

    def test_whitespace_around_the_indirection_is_tolerated(self, monkeypatch):
        monkeypatch.setenv("HELPER_TEST_KEY", "sk-from-env")
        assert resolve_api_key("  os.environ/HELPER_TEST_KEY  ") == "sk-from-env"

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_absent_key_becomes_an_empty_string(self, value):
        assert resolve_api_key(value) == ""

    def test_missing_variable_names_the_variable_and_how_to_set_it(self, monkeypatch):
        monkeypatch.delenv("HELPER_TEST_MISSING", raising=False)
        with pytest.raises(ValueError) as excinfo:
            resolve_api_key("os.environ/HELPER_TEST_MISSING")
        message = str(excinfo.value)
        assert "HELPER_TEST_MISSING" in message
        assert "export HELPER_TEST_MISSING=" in message

    def test_missing_variable_does_not_chain_a_confusing_key_error(self, monkeypatch):
        # `from None`: a KeyError on __cause__ reads like an internal bug rather
        # than a configuration problem the caller can fix.
        monkeypatch.delenv("HELPER_TEST_MISSING", raising=False)
        with pytest.raises(ValueError) as excinfo:
            resolve_api_key("os.environ/HELPER_TEST_MISSING")
        assert excinfo.value.__cause__ is None
