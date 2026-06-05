"""Generate Kavora sample narratives from canonical tags.

The script supports a local test mode for fast deterministic checks and a real
mode that uses a LangChain Agent backed by Anthropic Claude.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


MIN_WORDS = 200
MAX_WORDS = 400
DEFAULT_OUTPUT = "narratives.json"
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
REQUIRED_TAG_COLUMNS = (
    "Canonical Tag",
    "Category",
    "Severity",
    "Outcome",
    "Recommendation",
)

SYSTEM_PROMPT = """You are Kavora Narrative Generator, a careful synthetic data writer.
Generate realistic but fictional sample narratives for semantic classification testing.
Never include real people, real account numbers, real medical identifiers, or real case numbers.
Return only the narrative text requested by the user, with no title, markdown, bullets, or JSON."""

COMMON_WORDS = [
    "review",
    "document",
    "team",
    "request",
    "update",
    "summary",
    "context",
    "records",
    "timeline",
    "approval",
    "analysis",
    "details",
    "internal",
    "external",
    "policy",
    "reference",
    "question",
    "response",
    "draft",
    "issue",
    "followup",
    "evidence",
    "discussion",
    "process",
    "control",
]

CATEGORY_WORDS = {
    "Legal": [
        "counsel",
        "agreement",
        "privilege",
        "claim",
        "matter",
        "filing",
        "exhibit",
        "contract",
        "deposition",
        "settlement",
        "liability",
        "discovery",
    ],
    "Financial": [
        "account",
        "transaction",
        "balance",
        "invoice",
        "payment",
        "ledger",
        "portfolio",
        "expense",
        "forecast",
        "audit",
        "revenue",
        "transfer",
    ],
    "Insurance": [
        "claim",
        "policyholder",
        "coverage",
        "adjuster",
        "premium",
        "loss",
        "deductible",
        "incident",
        "underwriting",
        "benefit",
        "estimate",
        "inspection",
    ],
    "Healthcare": [
        "patient",
        "clinical",
        "diagnosis",
        "treatment",
        "provider",
        "referral",
        "medication",
        "appointment",
        "chart",
        "screening",
        "symptom",
        "care",
    ],
    "Government": [
        "agency",
        "permit",
        "public",
        "program",
        "caseworker",
        "benefit",
        "application",
        "notice",
        "compliance",
        "hearing",
        "ordinance",
        "review",
    ],
}

SENTENCE_TEMPLATES = [
    "The {category} team prepared a {tone} {doc} after the {subject} raised a question about {focus}.",
    "Several {domain} details were compared against the {tag_term} record so the reviewer could understand the {risk} concern.",
    "A {role} noted that the {doc} should describe the timeline, the supporting {domain}, and the expected {outcome} decision.",
    "During the review, the narrative connected {focus} with {tag_term} indicators and explained why the matter required {recommendation}.",
    "The draft avoided personal identifiers while still describing how {subject}, {domain}, and {risk} influenced the final recommendation.",
    "Follow-up notes showed that the {category} request remained active because the {severity} severity rating changed the handling path.",
    "The sample text included ordinary workplace language, a clear sequence of events, and enough {domain} context for classifier testing.",
    "By the end of the entry, the reviewer understood why the {outcome} outcome matched the canonical tag and why further review was useful.",
]


@dataclass(frozen=True)
class TagInfo:
    """Normalized representation of one row from tags.csv."""

    tag: str
    category: str
    severity: str
    outcome: str
    recommendation: str


@dataclass(frozen=True)
class NarrativeRecord:
    """One generated narrative paired with its canonical tag."""

    tag: str
    category: str
    severity: str
    outcome: str
    recommendation: str
    word_count: int
    narrative: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sample Kavora narratives aligned with tags.csv."
    )
    parser.add_argument(
        "--mode",
        choices=("test", "real"),
        default="test",
        help="Use local random text generation or the Anthropic-backed LangChain Agent.",
    )
    parser.add_argument(
        "--tags-file",
        type=Path,
        default=Path("tags.csv"),
        help="CSV file containing canonical tags.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="JSON file to write.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Generate only the first N selected tags.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        help="Generate only this canonical tag. May be provided more than once.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for reproducible test-mode narratives.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Anthropic model name for real mode. Defaults to ANTHROPIC_MODEL or a Claude Sonnet alias.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="LLM temperature for real mode.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retry count for real-mode word count corrections.",
    )
    return parser.parse_args(argv)


def load_tags(path: Path) -> list[TagInfo]:
    if not path.exists():
        raise FileNotFoundError(f"Tags file not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = [column for column in REQUIRED_TAG_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Tags file missing required columns: {', '.join(missing)}")

        tags = [
            TagInfo(
                tag=row["Canonical Tag"].strip(),
                category=row["Category"].strip(),
                severity=row["Severity"].strip(),
                outcome=row["Outcome"].strip(),
                recommendation=row["Recommendation"].strip(),
            )
            for row in reader
            if row.get("Canonical Tag", "").strip()
        ]

    if not tags:
        raise ValueError(f"Tags file contains no canonical tags: {path}")
    return tags


def select_tags(tags: Sequence[TagInfo], requested: Sequence[str] | None, limit: int | None) -> list[TagInfo]:
    selected = list(tags)
    if requested:
        requested_set = set(requested)
        selected = [tag for tag in selected if tag.tag in requested_set]
        found = {tag.tag for tag in selected}
        missing = sorted(requested_set - found)
        if missing:
            raise ValueError(f"Requested tags not found: {', '.join(missing)}")

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer")
        selected = selected[:limit]

    if not selected:
        raise ValueError("No tags selected for generation")
    return selected


def tokenize_tag(tag: str) -> list[str]:
    chunks = re.split(r"[._\-\s]+", tag)
    tokens: list[str] = []
    for chunk in chunks:
        tokens.extend(
            match.group(0).lower()
            for match in re.finditer(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", chunk)
        )
    return [token for token in tokens if token]


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def generate_test_narrative(tag: TagInfo, rng: random.Random) -> str:
    tag_terms = tokenize_tag(tag.tag) or [tag.tag.lower()]
    domain_words = CATEGORY_WORDS.get(tag.category, COMMON_WORDS)
    vocabulary = COMMON_WORDS + domain_words + tag_terms
    target_words = rng.randint(230, 320)
    sentences: list[str] = []

    while word_count(" ".join(sentences)) < target_words:
        template = rng.choice(SENTENCE_TEMPLATES)
        sentence = template.format(
            category=tag.category.lower(),
            severity=tag.severity.lower(),
            outcome=tag.outcome.lower(),
            recommendation=tag.recommendation.lower(),
            tone=rng.choice(["careful", "routine", "detailed", "fictional", "plain-language"]),
            doc=rng.choice(["memo", "entry", "summary", "record", "case note"]),
            subject=rng.choice(["manager", "analyst", "coordinator", "specialist", "reviewer"]),
            focus=rng.choice(vocabulary),
            risk=rng.choice(["confidentiality", "classification", "handling", "routing", "review"]),
            role=rng.choice(["reviewer", "supervisor", "analyst", "coordinator", "case owner"]),
            domain=rng.choice(domain_words),
            tag_term=rng.choice(tag_terms),
        )
        sentences.append(sentence)

    return " ".join(sentences)


def build_real_agent(model_name: str, temperature: float) -> Any:
    try:
        from langchain.agents import create_agent
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise RuntimeError(
            "Real mode requires langchain and langchain-anthropic. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Real mode requires ANTHROPIC_API_KEY in the environment")

    model = ChatAnthropic(
        model=model_name,
        temperature=temperature,
        max_tokens=1200,
    )
    return create_agent(model=model, tools=[], system_prompt=SYSTEM_PROMPT)


def build_real_prompt(tag: TagInfo, previous_word_count: int | None = None) -> str:
    correction = ""
    if previous_word_count is not None:
        correction = (
            f"\nThe previous response was {previous_word_count} words. "
            f"Rewrite it so the narrative is between {MIN_WORDS} and {MAX_WORDS} words."
        )

    return f"""Create one fictional sample narrative for semantic classification testing.

Canonical tag: {tag.tag}
Category: {tag.category}
Severity: {tag.severity}
Outcome: {tag.outcome}
Recommendation: {tag.recommendation}

Requirements:
- Write between {MIN_WORDS} and {MAX_WORDS} words.
- Make the text pseudo-organic, realistic, and aligned with the canonical tag.
- Do not mention that this is synthetic data.
- Do not include markdown, bullets, headings, JSON, names, account numbers, or identifiers.
- Keep the text fictional and suitable for classifier validation.{correction}"""


def extract_message_text(response: Any) -> str:
    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        messages = response.get("messages")
        if messages:
            last_message = messages[-1]
            content = getattr(last_message, "content", last_message)
            return content_to_text(content)
        output = response.get("output") or response.get("content")
        if output:
            return content_to_text(output)

    content = getattr(response, "content", response)
    return content_to_text(content)


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(getattr(item, "text", item)))
        return "\n".join(part for part in parts if part)
    return str(content)


def clean_narrative(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:text|json|markdown)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip().strip('"').strip()
    return re.sub(r"\s+", " ", cleaned)


def generate_real_narrative(agent: Any, tag: TagInfo, max_retries: int) -> str:
    last_word_count: int | None = None
    attempts = max_retries + 1

    for attempt in range(attempts):
        prompt = build_real_prompt(tag, last_word_count if attempt else None)
        response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        narrative = clean_narrative(extract_message_text(response))
        last_word_count = word_count(narrative)
        if MIN_WORDS <= last_word_count <= MAX_WORDS:
            return narrative

    raise RuntimeError(
        f"Claude returned {last_word_count} words for {tag.tag}; "
        f"expected {MIN_WORDS}-{MAX_WORDS} after {attempts} attempts."
    )


def generate_records(tags: Iterable[TagInfo], args: argparse.Namespace) -> list[NarrativeRecord]:
    records: list[NarrativeRecord] = []
    rng = random.Random(args.seed)
    agent = build_real_agent(args.model, args.temperature) if args.mode == "real" else None

    for index, tag in enumerate(tags, start=1):
        if args.mode == "real":
            print(f"[{index}] Generating {tag.tag} with {args.model}...", file=sys.stderr)
            narrative = generate_real_narrative(agent, tag, args.max_retries)
        else:
            narrative = generate_test_narrative(tag, rng)

        records.append(
            NarrativeRecord(
                tag=tag.tag,
                category=tag.category,
                severity=tag.severity,
                outcome=tag.outcome,
                recommendation=tag.recommendation,
                word_count=word_count(narrative),
                narrative=narrative,
            )
        )
    return records


def write_output(records: Sequence[NarrativeRecord], args: argparse.Namespace) -> None:
    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "model": args.model if args.mode == "real" else None,
            "source_tags_file": str(args.tags_file),
            "narrative_count": len(records),
            "word_count_range": {"min": MIN_WORDS, "max": MAX_WORDS},
        },
        "narratives": [asdict(record) for record in records],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        tags = load_tags(args.tags_file)
        selected_tags = select_tags(tags, args.tags, args.limit)
        records = generate_records(selected_tags, args)
        write_output(records, args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(records)} narratives to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())