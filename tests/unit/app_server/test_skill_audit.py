"""Tests for the skill content-addressing and drift audit.

Exercises both the standalone audit primitives and the wiring into the
existing skills router (``_load_skills_from_dir`` digesting + the ``/audit``
endpoint behaviour), proving the integration end to end.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from openhands.app_server.user import skills_router
from openhands.app_server.user.skill_audit import (
    audit_skill_directories,
    content_digest,
    jaccard_similarity,
    token_set,
)

# A pair of definitions: identical content -> same digest; a near-identical
# third -> drift; an unrelated fourth -> neither.
_SKILL_A = """---
name: planner
type: knowledge
---
Break the task into small verifiable steps and run the tests after each one.
"""

_SKILL_A_COPY = _SKILL_A  # byte-identical clone

_SKILL_A_DRIFT = """---
name: planner
type: knowledge
---
Break the task into small verifiable steps and run the linter after each one.
"""

_SKILL_UNRELATED = """---
name: weather
type: knowledge
---
Always greet the user warmly before answering any meteorology question.
"""


def _write(dir_path: Path, name: str, content: str) -> None:
    (dir_path / name).write_text(content, encoding='utf-8')


def test_content_digest_is_stable_and_distinguishing():
    assert content_digest(_SKILL_A) == content_digest(_SKILL_A_COPY)
    assert content_digest(_SKILL_A) != content_digest(_SKILL_A_DRIFT)


def test_jaccard_similarity_bounds():
    a = token_set('alpha beta gamma')
    assert jaccard_similarity(a, a) == 1.0
    assert jaccard_similarity(a, token_set('delta epsilon')) == 0.0


def test_audit_detects_duplicates_and_drift():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write(d, 'a.md', _SKILL_A)
        _write(d, 'a_copy.md', _SKILL_A_COPY)
        _write(d, 'a_drift.md', _SKILL_A_DRIFT)
        _write(d, 'unrelated.md', _SKILL_UNRELATED)
        _write(d, 'README.md', '# ignore me')

        report = audit_skill_directories([(d, 'global')], drift_threshold=0.8)

    # README.md is skipped; four real definitions audited.
    assert report.total_skills == 4

    # a.md and a_copy.md share one SHA-256 digest -> one duplicate group.
    assert len(report.duplicate_groups) == 1
    group = report.duplicate_groups[0]
    assert sorted(group.members) == ['global:a.md', 'global:a_copy.md']

    # a.md vs a_drift.md are near-identical but not byte-identical -> drift.
    drift = {(p.left, p.right) for p in report.drift_pairs}
    assert ('global:a.md', 'global:a_drift.md') in drift
    assert all(p.similarity >= 0.8 for p in report.drift_pairs)


def test_load_skills_from_dir_attaches_digest():
    """The existing loader now content-addresses each definition."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write(d, 'a.md', _SKILL_A)

        skills = skills_router._load_skills_from_dir(d, 'global')

    assert len(skills) == 1
    assert skills[0].digest == content_digest(_SKILL_A)


def test_audit_endpoint_uses_loaded_skill_dirs(monkeypatch):
    """The /audit endpoint audits exactly the router's configured dirs."""
    import asyncio

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write(d, 'a.md', _SKILL_A)
        _write(d, 'a_copy.md', _SKILL_A_COPY)

        empty = Path(tmp) / 'no_user_skills_here'
        monkeypatch.setattr(skills_router, 'GLOBAL_SKILLS_DIR', d)
        monkeypatch.setattr(skills_router, 'USER_SKILLS_DIR', empty)

        report = asyncio.run(skills_router.audit_skills(drift_threshold=0.85))

    assert report.total_skills == 2
    assert report.unique_digests == 1
    assert len(report.duplicate_groups) == 1
