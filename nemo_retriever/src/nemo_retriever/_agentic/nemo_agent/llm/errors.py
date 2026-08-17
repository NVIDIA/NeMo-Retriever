# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exception hierarchy for LLM backends.

Contract
--------
``BaseLLMBackend.completion`` / ``acompletion`` raise only :class:`LLMCallError`
subclasses **for call failures**. Any other exception escaping them is, by
definition, a bug (in the backend implementation or in the caller's inputs) and
must propagate untranslated so it surfaces as one.

Rules for backend implementations:

- Translate provider exceptions at the raise site with ``raise <LibraryError>(...)
  from e`` so the original exception stays chained on ``__cause__``.
- Wrap ONLY the client call in the translating ``try/except`` — never your own
  request-prep or response-parsing code. Errors from the wire get translated;
  errors from your code propagate raw.
- Never catch or wrap ``BaseException`` (``asyncio.CancelledError``,
  ``KeyboardInterrupt``); wrapping ``CancelledError`` breaks task cancellation.
- Add a new subclass here only when a caller *branches* on it. Nicer error
  messages alone do not justify a class — format the chained ``__cause__``.
"""


class LLMCallError(Exception):
    """An LLM API call failed.

    Instantiable catch-all for call failures that don't fit a more specific
    subclass (auth errors, timeouts, 5xx, unrecognized bad requests, ...).
    The provider exception, when there is one, is chained on ``__cause__``.
    """


class ContextLimitError(LLMCallError):
    """The request exceeded the model's context window.

    Also raised for indirect symptoms of an oversized prompt, e.g. a provider
    rejecting the request because the remaining completion budget is negative
    ("max_tokens must be at least 1, got -N").
    """


class ContentPolicyError(LLMCallError):
    """The provider refused the request on content-policy grounds."""


class RateLimitError(LLMCallError):
    """The provider rate-limited the request.

    ``BaseLLMBackend``'s public templates catch this and retry with a pause
    (see ``rate_limit_max_retries`` / ``rate_limit_retry_sleep_s`` on the
    config). Backends may set ``retry_after`` (seconds) when the provider
    communicates one; the retry loop honors it over the configured sleep,
    capped by a hardcoded ceiling (``base_backend._RETRY_AFTER_CAP_S``).
    """

    def __init__(self, message: str = "", retry_after: "float | None" = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after
