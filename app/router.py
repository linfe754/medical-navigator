from enum import Enum
import re
from openai import OpenAI


class Intent(str, Enum):
    NAVIGATION = "navigation"
    CLINICAL = "clinical"
    EMERGENCY = "emergency"
    OUT_OF_SCOPE = "out_of_scope"
    SOCIAL = "social"
    CLARIFY = "clarify"
    UNKNOWN = "unknown"


SOCIAL_PATTERNS = [
    r"^\s*(thanks|thank you|thank you very much|cheers|thx)\s*[!.]*$",
    r"^\s*(谢谢|謝謝|多谢|多謝|感谢|感謝)\s*[！!。.]*$",
    r"^\s*(hi|hello|hey)\s*[!.]*$",
    r"^\s*(你好|您好|嗨)\s*[！!。.]*$",
    r"^\s*(bye|goodbye|see you)\s*[!.]*$",
    r"^\s*(再见|再見|拜拜)\s*[！!。.]*$",
]


EMERGENCY_PATTERNS = [
    r"\bcall\s*000\b",
    r"\bemergency\b",
    r"\bambulance\b",
    r"\bcan't breathe\b",
    r"\bcannot breathe\b",
    r"\bnot breathing\b",
    r"\bunconscious\b",
    r"\bsevere chest pain\b",
    r"无法呼吸",
    r"不能呼吸",
    r"昏迷",
    r"叫救护车",
    r"叫救護車",
    r"严重.*胸痛",
    r"嚴重.*胸痛",
]


CLINICAL_PATTERNS = [
    r"\bdiagnos",
    r"\bwhat.*wrong with me\b",
    r"\bwhat.*disease\b",
    r"\bwhat.*condition\b",
    r"\bwhat medication\b",
    r"\bwhat medicine\b",
    r"\bwhat should i take\b",
    r"\bdosage\b",
    r"\bdose\b",
    r"\binterpret.*(test|result|scan|blood)\b",
    r"什么病",
    r"什麼病",
    r"怎么治疗",
    r"怎麼治療",
    r"吃什么药",
    r"吃什麼藥",
    r"用什么药",
    r"用什麼藥",
    r"药量",
    r"藥量",
    r"剂量",
    r"劑量",
    r"检查结果",
    r"檢查結果",
]


OUT_OF_SCOPE_PATTERNS = [
    r"\btell me (a )?joke\b",
    r"\bweather\b",
    r"\bsports?\b",
    r"\bfootball\b",
    r"\bsoccer\b",
    r"\bstock (price|market)\b",
    r"\bbitcoin\b",
    r"\bwrite (me )?(a )?(poem|story)\b",
    r"讲.*笑话",
    r"講.*笑話",
    r"天气",
    r"天氣",
    r"足球",
    r"股票",
    r"比特币",
    r"比特幣",
]


NAVIGATION_PATTERNS = [
    r"\bmedicare\b",
    r"\bgp\b",
    r"\bdoctor\b",
    r"\bhospital\b",
    r"\bpharmac",
    r"\bappointment\b",
    r"\breferral\b",
    r"\bbulk bill",
    r"\bhealthdirect\b",
    r"\bhealth service\b",
    r"\burgent care\b",
    r"\bfind.*(doctor|gp|hospital|pharmacy|clinic)\b",
    r"医保",
    r"醫保",
    r"医生",
    r"醫生",
    r"医院",
    r"醫院",
    r"药房",
    r"藥房",
    r"预约",
    r"預約",
    r"转诊",
    r"轉診",
    r"诊所",
    r"診所",
    r"医疗服务",
    r"醫療服務",
]


def matches(message: str, patterns: list[str]) -> bool:
    return any(
        re.search(pattern, message, re.IGNORECASE)
        for pattern in patterns
    )


def route_message(message: str) -> Intent:
    text = message.strip()

    if not text:
        return Intent.OUT_OF_SCOPE

    if matches(text, SOCIAL_PATTERNS):
        return Intent.SOCIAL

    if matches(text, EMERGENCY_PATTERNS):
        return Intent.EMERGENCY

    if matches(text, CLINICAL_PATTERNS):
        return Intent.CLINICAL

    if matches(text, OUT_OF_SCOPE_PATTERNS):
        return Intent.OUT_OF_SCOPE

    if matches(text, NAVIGATION_PATTERNS):
        return Intent.NAVIGATION

    return Intent.UNKNOWN


def classify_unknown(message: str, client: OpenAI) -> Intent:
    response = client.responses.create(
        model="gpt-5-nano",
        instructions="""
Classify the user's message by its explicit intent.

Return exactly ONE label:

navigation
clinical
emergency
out_of_scope
clarify

navigation:
The message explicitly asks about accessing, finding, understanding, or using
Australian healthcare services or the healthcare system.
Examples include healthcare providers, GP services, hospitals, pharmacies,
appointments, referrals, Medicare, healthcare costs, or where/how to access care.

clinical:
The message explicitly asks for diagnosis, interpretation of symptoms,
treatment, medication, test-result interpretation, prognosis, or personalised
medical advice.

emergency:
The message explicitly describes immediate danger, a possible medical emergency,
or asks for emergency medical help.

out_of_scope:
The message has a clear intent that is unrelated to healthcare navigation.
Examples include weather, jokes, coding, writing, sport, finance, or general knowledge.

clarify:
The message does not contain enough information to determine its intent.
Use clarify when the user asks for help but does not say what the help is about.

Critical rules:

- Classify only from information explicitly present in the user's message.
- Do not infer missing healthcare context.
- Do not assume the user is asking about healthcare merely because this classifier
  is used by a healthcare application.
- If healthcare intent is not explicitly established and there is no clear
  non-healthcare intent, return clarify.
- When uncertain between navigation and clarify, return clarify.
- When uncertain between clinical and navigation, use clinical only when the user
  is asking for clinical interpretation or advice.
- Do not answer the user's question.
- Do not explain the classification.
- Return only the label.

Examples:

"I don't know where to get help" -> clarify
"Can you help me?" -> clarify
"Where should I go?" -> clarify
"I don't know what to do" -> clarify

"My mum is unwell and I don't know where to take her" -> navigation
"I need to find a doctor" -> navigation
"Where can I get medical help tonight?" -> navigation

"My head has hurt for three days. What could it be?" -> clinical
"What medicine should I take?" -> clinical

"I can't breathe and need help now" -> emergency

"Tell me about tomorrow's weather" -> out_of_scope
"Help me write a cover letter" -> out_of_scope
""",
        input=message,
    )

    label = response.output_text.strip().lower()

    mapping = {
        "navigation": Intent.NAVIGATION,
        "clinical": Intent.CLINICAL,
        "emergency": Intent.EMERGENCY,
        "out_of_scope": Intent.OUT_OF_SCOPE,
        "clarify": Intent.CLARIFY,
    }

    return mapping.get(label, Intent.CLARIFY)

def is_service_finder_request(message: str) -> bool:
    text = message.lower()

    service_terms = [
        "gp",
        "doctor",
        "pharmacy",
        "pharmacist",
        "hospital",
        "urgent care",
        "clinic",
        "health service",
    ]

    find_terms = [
        "find",
        "near me",
        "nearby",
        "nearest",
        "closest",
        "where",
        "looking for",
    ]

    return (
        any(term in text for term in service_terms)
        and any(term in text for term in find_terms)
    )