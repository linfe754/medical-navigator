import os

import pytest
from dotenv import load_dotenv
from openai import OpenAI

from app.router import Intent, classify_unknown


load_dotenv()

pytestmark = pytest.mark.live

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@pytest.mark.parametrize(
    "message,expected",
    [
        (
            "I don't know where to get help",
            Intent.CLARIFY,
        ),
        (
            "Can you help me?",
            Intent.CLARIFY,
        ),
        (
            "My mum is unwell and I am not sure where to take her",
            Intent.NAVIGATION,
        ),
        (
            "Where can I get help tonight if my GP is closed?",
            Intent.NAVIGATION,
        ),
        (
            "My son has been coughing for three days. What could it be?",
            Intent.CLINICAL,
        ),
        (
            "I feel really sick, what do you think is wrong with me?",
            Intent.CLINICAL,
        ),
        (
            "I can't breathe and need help now",
            Intent.EMERGENCY,
        ),
        (
            "帮我写一封求职信",
            Intent.OUT_OF_SCOPE,
        ),
        (
            "我妈妈身体不舒服，我不知道应该去哪里看",
            Intent.NAVIGATION,
        ),
        (
            "我头痛三天了，可能是什么病？",
            Intent.CLINICAL,
        ),
        ("I really don't know where to get help", Intent.CLARIFY),
        ("I don't know who can help me", Intent.CLARIFY),
        ("Can someone please help me?", Intent.CLARIFY),
        ("I'm lost and need some help", Intent.CLARIFY),
        ("Where should I go?", Intent.CLARIFY),
        ("I need some advice", Intent.CLARIFY),

        ("I don't know where to get medical help", Intent.NAVIGATION),
        ("I need help finding a GP", Intent.NAVIGATION),
        ("My child is unwell and I don't know where to take him", Intent.NAVIGATION),
        ("I need somewhere to see a doctor tonight", Intent.NAVIGATION),
    ],
)
def test_live_classifier(message, expected):
    assert classify_unknown(message, client) == expected