"""Scripted conversations: plant facts, bury them under filler, then ask.

Filler turns paste long notes and ask for a one-word reply, so the context
grows quickly (forcing compaction) while output stays cheap.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Turn:
    kind: str  # plant | filler | question | control
    text: str
    expect: list[list[str]] = field(default_factory=list)


@dataclass(frozen=True)
class Scenario:
    name: str
    turns: list[Turn]

    @property
    def questions(self) -> list[Turn]:
        return [t for t in self.turns if t.kind in ("question", "control")]


_SUBJECTS = ["the onboarding flow", "the search index", "quarterly planning", "the design review",
             "customer interviews", "the analytics pipeline", "vendor contracts", "the help centre",
             "accessibility testing", "the marketing site", "localization", "the support rota"]
_VERBS = ["reviewed", "discussed", "raised concerns about", "proposed changes to", "agreed to revisit",
          "shared findings on", "asked for estimates on", "walked through", "prioritised"]
_TAILS = ["before the next sprint", "with the wider team", "pending feedback from users",
          "once the budget is confirmed", "after the holiday freeze", "in a follow-up meeting",
          "with no blockers noted", "as a low-priority item", "alongside the roadmap update"]
_PEOPLE = ["the product team", "engineering", "the support leads", "finance", "the data group",
           "two contractors", "the QA team", "leadership"]


def filler_notes(seed: int, paragraphs: int = 7) -> str:
    """Deterministic meeting notes with no facts the questions ask about."""
    rng = random.Random(seed)
    out = []
    for _ in range(paragraphs):
        sentences = [
            f"{rng.choice(_PEOPLE).capitalize()} {rng.choice(_VERBS)} {rng.choice(_SUBJECTS)} "
            f"{rng.choice(_TAILS)}."
            for _ in range(rng.randint(4, 6))
        ]
        out.append(" ".join(sentences))
    return "\n\n".join(out)


def _fillers(prefix: str, count: int, seed: int) -> list[Turn]:
    return [
        Turn("filler", f"Here are {prefix} notes, part {i + 1} of {count}. Reply with just "
                       f"'Noted.' and nothing else.\n\n{filler_notes(seed + i)}")
        for i in range(count)
    ]


LATE_FEE_CODE = '''def compute_late_fee(days_overdue: int, daily_rate: float = 0.75) -> float:
    grace_days = 3
    billable = max(0, days_overdue - grace_days)
    return round(min(billable * daily_rate, 30.0), 2)'''

SKU_CODE = '''def normalize_sku(raw: str) -> str:
    cleaned = raw.strip().upper().replace(" ", "-")
    return cleaned[:14] if len(cleaned) > 14 else cleaned.ljust(6, "0")'''

BILLING = Scenario("billing-service", [
    Turn("plant", "Please keep this function for later; reply with just 'Saved.'\n\n```python\n"
                  f"{LATE_FEE_CODE}\n```"),
    Turn("plant", "Note for later: our public API rate limit is 1,237 requests per minute per "
                  "key, with a burst allowance of 85. Reply with just 'Noted.'"),
    Turn("plant", "Team note: Priya Raman is the release manager and Tomasz Wojcik owns on-call. "
                  "Reply with just 'Noted.'"),
    Turn("plant", "Decision: we picked PostgreSQL over MongoDB because we need row-level security "
                  "for tenant isolation. Reply with just 'Noted.'"),
    *_fillers("billing sync", 14, seed=100),
    Turn("question", "Write out the exact compute_late_fee function I gave you earlier.",
         [["grace_days = 3"], ["billable * daily_rate"], ["30.0"]]),
    Turn("question", "What are our API rate limit and burst allowance? Just the numbers.",
         [["1237"], ["85"]]),
    Turn("question", "Who is the release manager?", [["priya raman"]]),
    Turn("question", "Why did we choose PostgreSQL over MongoDB?",
         [["row-level security", "row level security", "rls"]]),
    Turn("control", "What did we decide about the name of the Kubernetes cluster?"),
])

MOBILE = Scenario("mobile-app", [
    Turn("plant", "Keep this helper for later; reply with just 'Saved.'\n\n```python\n"
                  f"{SKU_CODE}\n```"),
    Turn("plant", "Spec note: the maximum upload size is 48 MB and thumbnails are 312 px wide. "
                  "Reply with just 'Noted.'"),
    Turn("plant", "Team note: Amara Okafor is the design lead for the mobile app. "
                  "Reply with just 'Noted.'"),
    Turn("plant", "Decision: offline mode ships in v2.3 because field agents lose signal on site. "
                  "Reply with just 'Noted.'"),
    *_fillers("mobile standup", 14, seed=200),
    Turn("question", "Show me the exact normalize_sku function from earlier.",
         [["[:14]"], ['ljust(6, "0")', "ljust(6, '0')"]]),
    Turn("question", "What are the maximum upload size and the thumbnail width?",
         [["48"], ["312"]]),
    Turn("question", "Who is the design lead?", [["amara okafor"]]),
    Turn("question", "Which version ships offline mode, and why?",
         [["2.3"], ["signal"]]),
    Turn("control", "What budget did we approve for the TV advertising campaign?"),
])

SCENARIOS = {s.name: s for s in (BILLING, MOBILE)}
