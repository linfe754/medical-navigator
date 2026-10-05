from typing import Literal

from pydantic import BaseModel


Intent = Literal[
    "navigation",
    "clinical",
    "emergency",
    "clarify",
    "out_of_scope",
    "social",
]

Language = Literal["en", "zh"]
Difficulty = Literal["straightforward", "boundary", "adversarial"]
Severity = Literal["normal", "high", "critical"]


class ExpectedBehaviour(BaseModel):
    intent: Intent
    tool: str | None = None
    required_text: list[str] = []
    forbidden_text: list[str] = []
    forbidden_behaviours: list[str] = []


class EvalCase(BaseModel):
    id: str
    category: str
    language: Language
    difficulty: Difficulty
    input: str
    expected: ExpectedBehaviour
    severity: Severity
    tags: list[str] = []
