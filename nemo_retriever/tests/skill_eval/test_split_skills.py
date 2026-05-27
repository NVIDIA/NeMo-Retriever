# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from nemo_retriever.skill_eval.dataset import DatasetEntry, _normalize_slash_command
from nemo_retriever.skill_eval.runner import _copy_skills, _render_prompt, _render_setup_prompt

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_skill(root: Path, name: str) -> None:
    skill_dir = root / name
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\n---\n\n# {name}\n", encoding="utf-8")
    (skill_dir / "PITFALLS.md").write_text("# Pitfalls\n", encoding="utf-8")
    (skill_dir / "references" / "REFERENCE.md").write_text("# Reference\n", encoding="utf-8")


def test_copy_skills_copies_split_skill_tree(tmp_path: Path) -> None:
    source = tmp_path / "skills"
    _write_skill(source, "nemo-retriever-ingest")
    _write_skill(source, "nemo-retriever-query")

    dest = tmp_path / "workdir" / ".claude" / "skills"

    _copy_skills(source, dest)

    assert (dest / "nemo-retriever-ingest" / "SKILL.md").is_file()
    assert (dest / "nemo-retriever-ingest" / "PITFALLS.md").is_file()
    assert (dest / "nemo-retriever-ingest" / "references" / "REFERENCE.md").is_file()
    assert (dest / "nemo-retriever-query" / "SKILL.md").is_file()


def test_copy_skills_accepts_single_skill_directory(tmp_path: Path) -> None:
    _write_skill(tmp_path, "nemo-retriever-query")

    dest = tmp_path / "dest"

    _copy_skills(tmp_path / "nemo-retriever-query", dest)

    assert (dest / "nemo-retriever-query" / "PITFALLS.md").is_file()


def test_copy_skills_follows_compatibility_symlinks(tmp_path: Path) -> None:
    package_skills = tmp_path / "package" / "skills"
    _write_skill(package_skills, "nemo-retriever-ingest")

    claude_skills = tmp_path / ".claude" / "skills"
    claude_skills.mkdir(parents=True)
    (claude_skills / "nemo-retriever-ingest").symlink_to(
        package_skills / "nemo-retriever-ingest",
        target_is_directory=True,
    )

    dest = tmp_path / "dest"

    _copy_skills(claude_skills, dest)

    assert (dest / "nemo-retriever-ingest" / "SKILL.md").is_file()
    assert not (dest / "nemo-retriever-ingest").is_symlink()


def test_copy_skills_accepts_mixed_symlink_and_local_skill_dirs(tmp_path: Path) -> None:
    package_skills = tmp_path / "package" / "skills"
    _write_skill(package_skills, "nemo-retriever-query")

    root_skills = tmp_path / ".agents" / "skills"
    root_skills.mkdir(parents=True)
    (root_skills / "nemo-retriever-query").symlink_to(
        package_skills / "nemo-retriever-query",
        target_is_directory=True,
    )
    _write_skill(root_skills, "contributor-workflow")
    (root_skills / "notes").mkdir()

    dest = tmp_path / "dest"

    _copy_skills(root_skills, dest)

    assert (dest / "nemo-retriever-query" / "SKILL.md").is_file()
    assert (dest / "contributor-workflow" / "SKILL.md").is_file()
    assert not (dest / "notes").exists()


def test_slash_prompts_use_task_specific_skill_names() -> None:
    entry = DatasetEntry(
        entry_id=1,
        query_id="q1",
        taxonomy_slot_id="retrieval",
        original_query="What was revenue in 2024?",
        paraphrased_prompt="Answer the revenue question.",
        ground_truth_pages=[],
    )

    assert _render_setup_prompt("c3_retriever_skill").strip() == "/nemo-retriever-ingest ./pdfs/"
    assert '/nemo-retriever-query "What was revenue in 2024?"' in _render_prompt(entry, "c3_retriever_skill")


def test_manifest_slash_aliases_rewrite_to_split_skills() -> None:
    assert _normalize_slash_command("/vidore-ingest ./pdfs/") == "/nemo-retriever-ingest ./pdfs/"
    assert _normalize_slash_command("/vidore What was revenue?") == "/nemo-retriever-query What was revenue?"
    assert _normalize_slash_command("/vidore_hr Find relevant pages") == "/nemo-retriever-query Find relevant pages"


def test_evaluate_skill_is_not_packaged_for_agents() -> None:
    for path in (
        REPO_ROOT / "nemo_retriever/src/nemo_retriever/.agents/skills/nemo-retriever-evaluate",
        REPO_ROOT / ".agents/skills/nemo-retriever-evaluate",
        REPO_ROOT / ".claude/skills/nemo-retriever-evaluate",
    ):
        assert not path.exists(), f"{path} should not be installed as an agent skill"
