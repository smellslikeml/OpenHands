"""Deterministic content-addressing and drift audit for skill definitions.

Skill markdown files (global and user microagents) steer the agent the same
way rule/agent-definition files do in other coding harnesses, yet that
configuration layer is largely unmanaged: identical definitions get copied
between installations and near-identical ones quietly diverge over time.

This module treats the enumerated skill files as a managed supply chain. It
content-addresses each definition with SHA-256 (so exact duplicates collapse
to one digest) and flags *prompt drift* — pairs that are highly similar but
not byte-identical — using deterministic Jaccard similarity over their token
sets. The audit is pure and tool-agnostic: same files in, same report out,
no LLM in the loop.

Adapted from "A Deterministic Control Plane for LLM Coding Agents"
(arXiv:2606.26924) — specifically its SHA-256 content-addressing and
Jaccard-based prompt-drift mechanisms.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel

# README files are listing scaffolding, not skill definitions; the skills
# router skips them when enumerating, and so do we.
_SKIP_FILENAMES = {'README.md'}

_TOKEN_RE = re.compile(r'\w+')


def content_digest(data: bytes | str) -> str:
    """Return the SHA-256 hex digest used to content-address a definition."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def token_set(text: str) -> frozenset[str]:
    """Tokenize ``text`` into a deterministic set of lowercased word tokens."""
    return frozenset(_TOKEN_RE.findall(text.lower()))


def jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard similarity of two token sets (0.0 when both are empty)."""
    if not left and not right:
        return 0.0
    union = len(left | right)
    if union == 0:
        return 0.0
    return len(left & right) / union


class SkillRecord(BaseModel):
    """A single skill definition reduced to its audit fingerprint."""

    identifier: str  # stable label, e.g. "global:planning/plan.md"
    source: str  # 'global' or 'user'
    digest: str  # SHA-256 of the raw file bytes
    token_count: int


class DuplicateGroup(BaseModel):
    """A set of definitions that share one SHA-256 digest (exact clones)."""

    digest: str
    members: list[str]


class DriftPair(BaseModel):
    """Two non-identical definitions whose similarity exceeds the threshold."""

    left: str
    right: str
    similarity: float


class SkillAuditReport(BaseModel):
    """Result of auditing a set of skill definitions."""

    total_skills: int
    unique_digests: int
    duplicate_groups: list[DuplicateGroup]
    drift_pairs: list[DriftPair]
    drift_threshold: float


def audit_skill_records(
    records: list[tuple[SkillRecord, frozenset[str]]],
    drift_threshold: float = 0.85,
) -> SkillAuditReport:
    """Audit pre-fingerprinted records for exact duplicates and drift.

    Args:
        records: ``(SkillRecord, token_set)`` pairs. The token set drives
            drift detection; the record's digest drives duplicate detection.
        drift_threshold: Minimum Jaccard similarity for a non-identical pair
            to count as drift. Higher = only very-close near-duplicates.

    Returns:
        A :class:`SkillAuditReport` with duplicate groups and drift pairs,
        both ordered deterministically.
    """
    by_digest: dict[str, list[str]] = {}
    for record, _ in records:
        by_digest.setdefault(record.digest, []).append(record.identifier)

    duplicate_groups = [
        DuplicateGroup(digest=digest, members=sorted(members))
        for digest, members in sorted(by_digest.items())
        if len(members) > 1
    ]

    drift_pairs: list[DriftPair] = []
    for i in range(len(records)):
        rec_i, tokens_i = records[i]
        for j in range(i + 1, len(records)):
            rec_j, tokens_j = records[j]
            if rec_i.digest == rec_j.digest:
                continue  # exact clone, already reported as a duplicate
            similarity = jaccard_similarity(tokens_i, tokens_j)
            if similarity >= drift_threshold:
                left, right = sorted((rec_i.identifier, rec_j.identifier))
                drift_pairs.append(
                    DriftPair(left=left, right=right, similarity=round(similarity, 4))
                )

    drift_pairs.sort(key=lambda p: (-p.similarity, p.left, p.right))

    return SkillAuditReport(
        total_skills=len(records),
        unique_digests=len(by_digest),
        duplicate_groups=duplicate_groups,
        drift_pairs=drift_pairs,
        drift_threshold=drift_threshold,
    )


def _fingerprint_file(
    md_file: Path, source: str, root: Path
) -> tuple[SkillRecord, frozenset[str]] | None:
    """Reduce one markdown file to a ``(SkillRecord, token_set)`` pair."""
    try:
        raw = md_file.read_bytes()
    except OSError:
        return None
    text = raw.decode('utf-8', errors='replace')
    tokens = token_set(text)
    try:
        rel = md_file.relative_to(root).as_posix()
    except ValueError:
        rel = md_file.name
    record = SkillRecord(
        identifier=f'{source}:{rel}',
        source=source,
        digest=content_digest(raw),
        token_count=len(tokens),
    )
    return record, tokens


def audit_skill_directories(
    directories: list[tuple[Path, str]],
    drift_threshold: float = 0.85,
) -> SkillAuditReport:
    """Enumerate skill markdown files and audit them.

    Mirrors the skills router's enumeration convention (recursive ``*.md``,
    skipping README files) so the audit covers exactly the same definitions
    the router lists.

    Args:
        directories: ``(path, source_label)`` pairs to scan.
        drift_threshold: Forwarded to :func:`audit_skill_records`.
    """
    records: list[tuple[SkillRecord, frozenset[str]]] = []
    for skills_dir, source in directories:
        if not skills_dir.exists():
            continue
        for md_file in sorted(skills_dir.rglob('*.md')):
            if md_file.name in _SKIP_FILENAMES:
                continue
            fingerprint = _fingerprint_file(md_file, source, skills_dir)
            if fingerprint is not None:
                records.append(fingerprint)
    return audit_skill_records(records, drift_threshold=drift_threshold)
