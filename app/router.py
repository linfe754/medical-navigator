from enum import Enum
import re
from openai import OpenAI
import time

from app.observability.telemetry import log_llm


class Intent(str, Enum):
    NAVIGATION = "navigation"
    CLINICAL = "clinical"
    EMERGENCY = "emergency"
    OUT_OF_SCOPE = "out_of_scope"
    SOCIAL = "social"
    CLARIFY = "clarify"
    UNKNOWN = "unknown"

class Topic(str, Enum):
    MEDICARE = "medicare"
    SERVICE_SEARCH = "service_search"
    TRANSPORT = "transport"
    HEALTHCARE_NAVIGATION = "healthcare_navigation"
    
SOCIAL_PATTERNS = [
    r"^\s*(hi|hello|hey|good morning|good afternoon|good evening)\s*[!.]*$",
    r"^\s*(你好|您好|嗨|早上好|下午好|晚上好)\s*[！!。.]*$",

    r"^\s*(thanks|thank you|thank you very much|many thanks|cheers|thx)(?:,\s*)?(?:that helped|that was helpful|for your help|so much)?\s*[!.]*$",
    r"^\s*(谢谢|謝謝|多谢|多謝|感谢|感謝)(?:你)?(?:，|,)?(?:帮了大忙|幫了大忙|非常有帮助|非常有幫助)?\s*[！!。.]*$",

    r"^\s*(bye|goodbye|bye for now|see you|see you later)\s*[!.]*$",
    r"^\s*(再见|再見|拜拜)\s*[！!。.]*$",
    r"^\s*(thanks|thank you|cheers).*?(that's all|that is all|nothing else|no more questions|all i needed).*?[!.]*$",
    r"^\s*(谢谢|謝謝).*?(没有别的问题|沒有別的問題|没别的问题|沒別的問題|就这些|就這些).*?[！!。.]*$",
    r"^\s*(cheers|thanks|thank you)(?:,\s*)?(?:thanks|thank you)?(?:\s+for\s+(?:your|the)\s+help)?\s*[!.]*$",
]


EMERGENCY_PATTERNS = [
    r"\bcall\s*000\b",
    r"\bambulance\b",
    r"\bcan't breathe\b",
    r"\bcannot breathe\b",
    r"\bnot breathing\b",
    r"\bstruggl(?:e|ing)\s+to\s+breathe\b",
    r"\bdifficulty breathing\b",
    r"\bshortness of breath\b",
    r"\bunconscious\b",
    r"\bsevere chest pain\b",
    r"\bcrushing chest pain\b",
    r"\boverdose\b",
    r"\btook too many (?:pills|tablets|medications|medicines|drugs)\b",
    r"\bswallowed too many (?:pills|tablets|medications|medicines|drugs)\b",
    r"\bwant to (?:kill|hurt) myself\b",
    r"\btrying to (?:kill|hurt) myself\b",
    r"\btried to (?:kill|hurt) myself\b",
    r"\bsuicidal\b",
    r"无法呼吸",
    r"不能呼吸",
    r"呼吸.*困难",
    r"呼吸.*困難",
    r"昏迷",
    r"叫救护车",
    r"叫救護車",
    r"严重.*胸痛",
    r"嚴重.*胸痛",
    r"剧烈.*胸痛",
    r"劇烈.*胸痛",
    r"舌头.*肿.*呼吸困难",
    r"舌頭.*腫.*呼吸困難",
    r"吞了很多药.*结束生命",
    r"吞了很多藥.*結束生命",
]


CLINICAL_PATTERNS = [
    r"\bdiagnos",
    r"\bwhat.*wrong with me\b",
    r"\bwhat.*disease\b",
    r"\bwhat.*condition\b",
    r"\bwhat medication\b",
    r"\bwhat medicine\b",
    r"\bwhat should i take\b",
    r"\bwhich medication\b",
    r"\bwhich medicine\b",
    r"\bwhich antibiotic\b",
    r"\bshould i take.*(?:medicine|medication|antibiotic)\b",
    r"\bdosage\b",
    r"\bdose\b",
    r"\binterpret.*(test|result|scan|blood)\b",
    r"\bwhich (?:type|kind) of (?:doctor|specialist) should i see for\b",
    r"什么病",
    r"什麼病",
    r"怎么治疗",
    r"怎麼治療",
    r"吃什么药",
    r"吃什麼藥",
    r"用什么药",
    r"用什麼藥",
    r"要吃.*药",
    r"要吃.*藥",
    r"抗生素",
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
    r"\b(?:stock|share|shares)\b.*\b(?:buy|sell|invest|investment)\b",
    r"\b(?:buy|sell|invest|investment)\b.*\b(?:stock|share|shares)\b",
    r"\bbitcoin\b",
    r"\bwrite (me )?(a )?(poem|story)\b",
    r"\b(?:write|draft|prepare|help me write).*\b(?:cover letter|resume|cv|job application)\b",
    r"\b(?:cover letter|resume|cv|job application)\b",
    r"讲.*笑话",
    r"講.*笑話",
    r"天气",
    r"天氣",
    r"足球",
    r"股票",
    r"比特币",
    r"比特幣",
    r"(?:写|寫|帮我写|幫我寫).*(?:求职信|求職信|简历|簡歷|申请信|申請信)",
    r"(?:求职信|求職信|简历|簡歷|申请信|申請信)",
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
    start = time.perf_counter()

    response = client.responses.create(
        model="gpt-5-nano",
        instructions="""
Classify the user's message according to the Medical Navigator product boundary.

Return exactly ONE label:

navigation
clinical
emergency
out_of_scope
clarify

Do not answer the user's question.
Do not explain your reasoning.
Return only the label.


INTENT DEFINITIONS

emergency:
The message contains signs of an immediate or potentially life-threatening
medical emergency, immediate danger, or an urgent request for emergency help.

Emergency signals include, but are not limited to:
- inability to breathe or severe difficulty breathing
- severe or crushing chest pain
- loss of consciousness
- possible stroke signs, such as sudden facial drooping, sudden speech
  difficulty, or sudden one-sided weakness
- severe allergic reaction affecting breathing
- serious overdose or poisoning
- self-harm or suicidal actions indicating immediate danger

An emergency signal takes priority over every other intent, even when the user
also asks about transport, hospitals, medication, or another healthcare service.


clinical:
The message asks for clinical judgement or personalised medical guidance.

This includes:
- diagnosis or possible diagnosis
- interpretation of symptoms
- deciding what a symptom or condition means
- treatment advice
- medication or dosage advice
- interpretation of tests, scans, blood results, or other clinical results
- prognosis
- deciding which type of doctor, specialist, department, or medical service
  is appropriate based on symptoms or a health condition

Choosing an appropriate type of care based on symptoms requires clinical
assessment and must therefore be classified as clinical.


navigation:
The message asks how to access, find, understand, or use Australian healthcare
services or the healthcare system without requiring clinical judgement.

This includes:
- finding a known type of healthcare provider or service
- finding a GP, hospital, pharmacy, specialist, or other healthcare facility
- directions or public transport to a healthcare service
- appointments and referrals
- Medicare
- healthcare costs or eligibility
- administrative information about accessing care

If the user already knows the provider, service, specialty, or facility they
want and asks where to find it, how to access it, or how to get there, classify
as navigation.

A user may mention an illness or symptom as context without making the request
clinical. Classify according to what the user is asking for, unless an emergency
signal is present.


out_of_scope:
The message has a clear intent unrelated to healthcare navigation or the
Medical Navigator product.

Examples include:
- weather
- jokes
- coding
- writing tasks
- employment applications
- sport
- investing or finance
- unrelated general knowledge

The presence of a healthcare word, hospital name, doctor name, or medical
organisation does not by itself make a request navigation.

For example, asking for investment advice about a hospital company or asking
for help writing a hospital job application is out_of_scope.


clarify:
The user's intended task cannot be determined from the message.

Use clarify when:
- the user asks for help but does not say what kind of help
- the message refers to missing prior context, such as "there", "that place",
  "it", or "them", and the current message alone does not establish what the
  reference means
- the message is too ambiguous to establish healthcare navigation, clinical,
  emergency, or a clear non-healthcare intent
- the message is meaningless or insufficient to determine an intent

Do not use clarify merely because information required to complete a known
navigation task is missing.

For example:
"Give me directions to the hospital, but I haven't chosen which hospital."
is navigation, not clarify. The navigation task is clear even though a required
destination is missing.


DECISION PRIORITY

Apply these rules in order:

1. If an emergency signal is present, return emergency.
2. Otherwise, if the request requires clinical judgement, return clinical.
3. Otherwise, if the request is clearly about healthcare access or navigation,
   return navigation.
4. Otherwise, if a clear non-healthcare task is present, return out_of_scope.
5. Otherwise, return clarify.


CRITICAL RULES

- Classify only from information present in the user's message.
- Do not invent or infer missing healthcare context.
- Do not assume healthcare intent merely because this classifier belongs to a
  healthcare application.
- Classify the user's requested task, not merely keywords appearing in the text.
- Historical or contextual mentions do not override the user's current task.
- A healthcare-related noun alone does not establish navigation intent.
- Emergency safety signals override the requested task.
- Do not answer the user's question.
- Do not provide medical advice.
- Do not explain the classification.
- Return exactly one label and nothing else.


BOUNDARY EXAMPLES

"I can't breathe and need help now"
-> emergency

"I have crushing chest pain. What tram goes to Royal Melbourne Hospital?"
-> emergency

"My dad suddenly has facial drooping and slurred speech."
-> emergency

"我爸突然嘴歪了，说话也不清楚。"
-> emergency

"孩子呼吸很困难，怎么去医院？"
-> emergency

"I took too many tablets and want to find a clinic."
-> emergency


"My head has hurt for three days. What could it be?"
-> clinical

"What medicine should I take?"
-> clinical

"Which type of doctor should I see for recurring headaches?"
-> clinical

"经常头疼应该看哪个科？"
-> clinical


"Find a neurologist near me."
-> navigation

"帮我找附近的神经科。"
-> navigation

"I need to find a doctor."
-> navigation

"Where can I get medical help tonight?"
-> navigation

"My mum is unwell and I need directions to Royal Melbourne Hospital."
-> navigation

"I was discharged from emergency yesterday. How do I get back to the hospital
for my follow-up appointment?"
-> navigation

"Give me directions to the hospital. I haven't chosen which hospital yet."
-> navigation


"Which hospital company shares should I buy?"
-> out_of_scope

"Write a cover letter for a receptionist job at Royal Melbourne Hospital."
-> out_of_scope

"Tell me about tomorrow's weather."
-> out_of_scope

"Help me write a cover letter."
-> out_of_scope


"I don't know where to get help."
-> clarify

"Can you help me?"
-> clarify

"Where should I go?"
-> clarify

"I don't know what to do."
-> clarify

"asdf qwer zxcv"
-> clarify

"Can you help me get there?"
-> clarify
""",
        input=message,
    )

    log_llm(
        operation="classify_unknown",
        model="gpt-5-nano",
        latency_ms=round(
            (time.perf_counter() - start) * 1000
        ),
        usage=response.usage,
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

def classify_intent(
    message: str,
    client: OpenAI,
) -> tuple[Intent, str]:
    intent = route_message(message)

    if intent != Intent.UNKNOWN:
        return intent, "regex"

    return classify_unknown(message, client), "nano"

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
    
def is_transport_request(message: str) -> bool:
    text = message.lower().strip()

    transport_patterns = [
        r"\bhow do i get to\b",
        r"\bhow do i get from\b",
        r"\bhow can i get to\b",
        r"\bhow to get to\b",
        r"\bpublic transport\b",
        r"\bby public transport\b",
        r"\bby tram\b",
        r"\bby bus\b",
        r"\bby train\b",
        r"\bnearest tram\b",
        r"\bnearest bus\b",
        r"\bnearest train\b",
        r"怎么去",
        r"如何去",
        r"公共交通",
        r"坐电车",
        r"坐公交",
        r"坐火车",
    ]

    return any(
        re.search(pattern, text)
        for pattern in transport_patterns
    )
    
    
def extract_transport_destination(
    message: str,
    client: OpenAI,
) -> str | None:
    start = time.perf_counter()

    response = client.responses.create(
        model="gpt-5-nano",
        instructions=(
            "Extract only the healthcare facility destination from the user's "
            "message. Return only the destination text, with no explanation. "
            "If no healthcare facility destination can be identified, return NONE."
        ),
        input=message,
    )

    log_llm(
        operation="extract_transport_destination",
        model="gpt-5-nano",
        latency_ms=round(
            (time.perf_counter() - start) * 1000
        ),
        usage=response.usage,
    )

    destination = response.output_text.strip()

    if not destination or destination.upper() == "NONE":
        return None

    return destination


def extract_transport_origin(
    message: str,
    client: OpenAI,
) -> str | None:
    start = time.perf_counter()

    response = client.responses.create(
        model="gpt-5-nano",
        instructions=(
            "Extract only the starting location from the user's public transport "
            "request. Return only the location text, with no explanation. "
            "If no starting location is explicitly provided, return NONE."
        ),
        input=message,
    )

    log_llm(
        operation="extract_transport_origin",
        model="gpt-5-nano",
        latency_ms=round(
            (time.perf_counter() - start) * 1000
        ),
        usage=response.usage,
    )

    origin = response.output_text.strip()

    if not origin or origin.upper() == "NONE":
        return None

    return origin

def detect_topic(message: str) -> Topic | None:
    text = message.lower().strip()

    if is_transport_request(text):
        return Topic.TRANSPORT

    if is_service_finder_request(text):
        return Topic.SERVICE_SEARCH

    if re.search(r"\bmedicare\b|医保|醫保", text):
        return Topic.MEDICARE

    if matches(text, NAVIGATION_PATTERNS):
        return Topic.HEALTHCARE_NAVIGATION

    return None

