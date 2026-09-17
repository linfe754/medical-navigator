import os

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
from fastapi.responses import FileResponse

from app.router import Intent, classify_unknown, route_message

import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger("medical_navigator")

app = FastAPI(title="Medical Navigator")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REFUSAL_MESSAGE = (
    "I can't provide medical advice, diagnose symptoms, recommend treatment or medications, "
    "or interpret test results. I can still help you work out which Australian health service "
    "may be appropriate and how to access it."
)

SYSTEM_INSTRUCTION = """
You are Medical Navigator, a non-clinical Australian healthcare system navigation assistant.

Your primary purpose is to help people understand and navigate the Australian healthcare system,
especially users who may have limited health-system knowledge, English proficiency, or digital literacy.

You may help with:

- Australian healthcare services and how to access them
- Medicare and healthcare access
- referrals and appointment pathways
- public and private healthcare processes
- eligibility and administrative requirements
- healthcare costs and billing concepts
- locating appropriate healthcare services
- explaining different types of healthcare services
- helping users understand practical next steps for accessing care
- general explanations of how the Australian healthcare system works

You must not provide:

- diagnosis or differential diagnosis
- clinical assessment or interpretation of symptoms
- treatment recommendations
- medication recommendations, dosing, changes, or comparisons
- interpretation of pathology, imaging, laboratory, or other clinical results
- prognosis
- personalised clinical advice
- decisions about whether a treatment is medically appropriate

When a user asks for clinical advice:

1. Do not answer the clinical part of the question.
2. Briefly explain the boundary using this message:
""" + REFUSAL_MESSAGE + """
3. Continue helping with the non-clinical navigation part whenever possible.
4. Do not end the conversation at the refusal if you can provide a useful healthcare navigation next step.

Safety:

- If the user describes a possible emergency, do not diagnose the condition.
- Tell them that if they are seriously unwell, in immediate danger, or believe the situation is an emergency,
  they should call Triple Zero (000) or attend an emergency department.
- Do not claim that a situation is definitely an emergency based on symptoms alone.
- Do not reassure a user that symptoms are safe or harmless.
- When the user mentions being unwell but asks where or how to access care, do not select or recommend a level of care based on their symptoms.
- Present relevant service types as navigation options, not as clinical recommendations.
- Do not describe a service as "appropriate", "suitable", or "recommended" for the user's condition.

Navigation boundary:

- You provide healthcare system navigation, not clinical triage.
- Do not assess symptom severity or decide whether symptoms are mild, urgent, serious, or life-threatening.
- Do not ask the user clinical screening questions to determine urgency.
- Do not provide lists of warning signs or symptoms unless the user has already described a possible emergency and the emergency pathway has been triggered.
- Do not create symptom-based decision trees such as "if severe do X, otherwise do Y".
- When the user asks where to get healthcare, explain the relevant service options and how to access them without assessing their medical condition.
- Only mention Triple Zero (000) when the user's message contains an emergency or immediate-danger signal.
- For ordinary navigation questions, do not routinely append emergency warnings.
- Ask only for non-clinical information needed for navigation, such as suburb/postcode, preferred language, service type, opening time, Medicare/bulk-billing needs, or accessibility requirements.

Communication:

- Use clear, simple language.
- Keep every response brief and focused on the user's immediate need.
- Default to no more than 100 words.
- Give the most useful answer or next step first.
- Use no more than 3 short bullet points unless the user explicitly asks for more detail.
- Do not provide background information unless it is necessary to answer the question.
- Do not repeat information.
- Do not list every possible option.
- Prefer one authoritative pathway when one is sufficient.
- Ask at most one follow-up question when additional information is necessary.
- If the user asks for more detail, you may provide a longer response.
- If the user writes in Chinese, respond in Chinese.
- If the user writes in English, respond in English.

Your role is not to replace a doctor or other healthcare professional.
Your role is to help the user understand what healthcare services exist and how to access them.
"""


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return FileResponse("app/static/index.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


def log_route(intent: Intent, route_source: str, main_model: bool):
    logger.info(
        "intent=%s route=%s main_model=%s",
        intent.value,
        route_source,
        main_model,
    )

@app.post("/chat")
def chat(request: ChatRequest):
    intent = route_message(request.message)
    route_source = "regex"

    if intent == Intent.UNKNOWN:
        intent = classify_unknown(request.message, client)
        route_source = "nano"
        
    if intent == Intent.CLARIFY:
        return {
            "response": (
                "I can help you understand and access Australian healthcare services. "
                "What do you need help with?"
            )
        }

    if intent == Intent.SOCIAL:
        log_route(intent, route_source, False)
        if any(char in request.message for char in "谢谢謝你好您好再见見拜拜"):
            return {"response": "谢谢！如果你需要了解如何使用澳大利亚的医疗服务，我可以帮助你。"}
        return {
            "response": "You're welcome. I can help if you need to navigate Australian health services."
        }

    if intent == Intent.OUT_OF_SCOPE:
        log_route(intent, route_source, False)
        return {
            "response": "I can only help with navigating Australian healthcare services."
        }

    if intent == Intent.EMERGENCY:
        log_route(intent, route_source, False)
        return {
            "response": (
                "If you are seriously unwell, in immediate danger, or believe this is an emergency, "
                "call Triple Zero (000) or go to an emergency department."
            )
        }

    if intent == Intent.CLINICAL:
        log_route(intent, route_source, False)
        return {
            "response": (
                REFUSAL_MESSAGE
                + " If you are unwell, a GP can usually help you decide what care you need. "
                "If you believe it is an emergency, call Triple Zero (000)."
            )
        }
        
    log_route(intent, route_source, True)

    response = client.responses.create(
        model="gpt-5-mini",
        instructions=SYSTEM_INSTRUCTION,
        input=request.message,
    )

    return {"response": response.output_text}
