# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nemo_retriever.harness.vidore_access import VidoreAccessError, check_vidore_access


class _Response:
    def __init__(
        self,
        status_code: int,
        *,
        final_host: str = "cas-bridge.xethub.hf.co",
        content: bytes = b"x",
    ) -> None:
        self.status_code = status_code
        self.history = [SimpleNamespace(status_code=302)]
        self.url = f"https://{final_host}/signed-object"
        self.headers = {"content-range": "bytes 0-0/1024"} if status_code == 206 else {}
        self.content = content
        self.iter_content_calls: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_content(self, chunk_size: int):
        self.iter_content_calls.append(chunk_size)
        if self.content:
            yield self.content[:chunk_size]


class _Api:
    def __init__(self) -> None:
        self.whoami_calls = 0

    def whoami(self):
        self.whoami_calls += 1
        return {"auth": {"accessToken": {"role": "read"}}}

    def dataset_info(self, repo_id: str, *, files_metadata: bool):
        assert repo_id == "vidore/vidore_v3_hr"
        assert files_metadata is True
        return SimpleNamespace(
            sha="abc123",
            private=False,
            gated=False,
            siblings=[
                SimpleNamespace(rfilename="queries/test-00000-of-00001.parquet"),
                SimpleNamespace(rfilename="qrels/test-00000-of-00001.parquet"),
                SimpleNamespace(rfilename="corpus/test-00000-of-00001.parquet"),
            ],
        )


def test_check_vidore_access_probes_one_byte_from_each_evaluation_partition() -> None:
    api = _Api()
    requests = []

    def get(url, **kwargs):
        response = _Response(206)
        requests.append((url, kwargs, response))
        return response

    results = check_vidore_access(
        dataset_names=("vidore_v3_hr",),
        token="hf-secret",
        timeout_seconds=7,
        api=api,
        get=get,
    )

    assert api.whoami_calls == 1
    assert len(results) == 1
    assert results[0].dataset == "vidore_v3_hr"
    assert results[0].revision == "abc123"
    assert results[0].checked_partitions == ("queries", "qrels", "corpus")
    assert results[0].authenticated is True
    assert len(requests) == 3
    assert all(call[1]["headers"]["Range"] == "bytes=0-0" for call in requests)
    assert all(call[1]["headers"]["Authorization"] == "Bearer hf-secret" for call in requests)
    assert all(call[1]["timeout"] == 7 for call in requests)
    assert all(call[1]["stream"] is True for call in requests)
    assert all(call[2].iter_content_calls == [1] for call in requests)


def test_check_vidore_access_reports_cas_redirect_failure_without_leaking_token() -> None:
    def get(url, **kwargs):
        return _Response(403)

    with pytest.raises(VidoreAccessError) as exc_info:
        check_vidore_access(
            dataset_names=("vidore_v3_hr",),
            token="hf-secret",
            api=_Api(),
            get=get,
        )

    message = str(exc_info.value)
    assert "vidore_v3_hr" in message
    assert "queries/test-00000-of-00001.parquet" in message
    assert "status chain 302 -> 403" in message
    assert "cas-bridge.xethub.hf.co" in message
    assert "hf-secret" not in message


def test_check_vidore_access_requires_queries_qrels_and_corpus() -> None:
    api = _Api()

    def incomplete_dataset_info(repo_id: str, *, files_metadata: bool):
        return SimpleNamespace(
            sha="abc123",
            private=False,
            gated=False,
            siblings=[SimpleNamespace(rfilename="queries/test-00000-of-00001.parquet")],
        )

    api.dataset_info = incomplete_dataset_info

    with pytest.raises(VidoreAccessError, match="missing evaluation partitions: qrels, corpus"):
        check_vidore_access(dataset_names=("vidore_v3_hr",), token="hf-secret", api=api, get=lambda *a, **k: None)


def test_check_vidore_access_missing_token_recommends_direct_export(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    with pytest.raises(VidoreAccessError, match="export HF_TOKEN"):
        check_vidore_access(dataset_names=("vidore_v3_hr",))


def test_check_vidore_access_rejects_an_empty_success_response() -> None:
    def get(url, **kwargs):
        return _Response(206, content=b"")

    with pytest.raises(VidoreAccessError, match="returned an empty response"):
        check_vidore_access(
            dataset_names=("vidore_v3_hr",),
            token="hf-secret",
            api=_Api(),
            get=get,
        )
