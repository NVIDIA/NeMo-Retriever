import pandas as pd
import pytest

from nemo_retriever.recall.core import _hit_to_audio_segment_key, _normalize_query_df, is_hit_at_k


@pytest.mark.parametrize(
    "match_mode,df,expected",
    [
        (
            "audio_segment",
            pd.DataFrame({"question": ["q1"], "name": ["clip.mp3"], "start_time": [12.0], "end_time": [15.0]}),
            ["clip\t12.000000\t15.000000"],
        ),
    ],
)
def test_normalize_query_df_modes(match_mode: str, df: pd.DataFrame, expected: list[str]) -> None:
    out = _normalize_query_df(df, match_mode=match_mode)
    assert out["golden_answer"].tolist() == expected


@pytest.mark.parametrize(
    "match_mode,golden,retrieved,k,expected",
    [
        (
            "audio_segment",
            "clip\t10.000000\t20.000000",
            ["clip\t11.000000\t13.000000", "other\t12.000000\t14.000000"],
            1,
            True,
        ),
        (
            "audio_segment",
            "clip\t10.000000\t20.000000",
            ["clip\t25.000000\t27.000000", "clip\t9.000000\t10.000000"],
            1,
            False,
        ),
    ],
)
def test_is_hit_at_k_modes(match_mode: str, golden: str, retrieved: list[str], k: int, expected: bool) -> None:
    assert is_hit_at_k(golden, retrieved, k, match_mode=match_mode) is expected


def test_hit_to_audio_segment_key_converts_millis_to_seconds() -> None:
    hit = {
        "source_id": "/tmp/sample_clip.mp3",
        "metadata": (
            "{'source_path': '/tmp/sample_clip.mp3', 'duration': 79.67, " "'segment_start': 320, 'segment_end': 4880}"
        ),
    }

    assert _hit_to_audio_segment_key(hit) == "sample_clip\t0.320000\t4.880000"


def test_hit_to_audio_segment_key_normalizes_content_metadata_times() -> None:
    hit = {
        "source_id": "/tmp/sample_clip.mp3",
        "metadata": (
            "{'source_path': '/tmp/sample_clip.mp3', 'duration': 79.67, "
            "'content_metadata': {'start_time': 320, 'end_time': 4880}}"
        ),
    }

    assert _hit_to_audio_segment_key(hit) == "sample_clip\t0.320000\t4.880000"
