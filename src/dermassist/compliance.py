"""Single source of truth for the compliance disclaimer.

Per HANDOFF.md, this string MUST appear on every UI surface, report, and README.
Import it from here everywhere rather than re-typing it, so there is exactly one
canonical wording.
"""

DISCLAIMER: str = (
    "For research and educational use only. Not for clinical diagnosis."
)

# Longer-form notice for prominent surfaces (README banner, UI header).
DISCLAIMER_LONG: str = (
    "⚠️ DermAssist is a research and educational tool, NOT a medical device. "
    "Its output is not a diagnosis and must never be used for clinical "
    "decision-making. Every analysis requires review by a qualified human; the "
    "human-review gate is a hard stop, never a rubber stamp."
)

__all__ = ["DISCLAIMER", "DISCLAIMER_LONG"]
