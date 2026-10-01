import os
import logging
import time
import uuid
import json
from app.observability.telemetry import (
    log_llm,
    log_route,
    log_stream,
    request_id_ctx,
)

from fastapi import FastAPI, Request

from dotenv import load_dotenv
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from app.memory import memory
from app.router import (
    Intent,
    classify_unknown,
    detect_topic,
    is_service_finder_request,
    is_transport_request,
    route_message,
)
from app.services.transport_service import (
    handle_pending_facility_selection,
    handle_transport_request,
)


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger("medical_navigator")


app = FastAPI(title="Medical Navigator")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request_id_ctx.set(request_id)
    start = time.perf_counter()

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        latency_ms = round((time.perf_counter() - start) * 1000)

        logger.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "latency_ms": latency_ms,
                }
            )
        )

app.mount("/static", StaticFiles(directory="app/static"), name="static")


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
- Do not generate public transport routes, stops, lines, travel times, or directions from your own knowledge.
- Public transport directions must only be provided from the transport tool.
- If transport information is requested but tool results are unavailable, ask the user to clarify the origin or destination instead of generating a route.

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
    session_id: str = "default"


@app.get("/")
def root():
    return FileResponse("app/static/index.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


def resolve_topic(session_id: str, message: str) -> str | None:
    explicit_topic = detect_topic(message)

    if explicit_topic:
        memory.set_topic(session_id, explicit_topic.value)
        return explicit_topic.value

    return memory.get_session(session_id).state.current_topic


@app.post("/chat")
def chat(request: ChatRequest):
    topic = resolve_topic(
        request.session_id,
        request.message,
    )

    pending_response = handle_pending_facility_selection(
        request,
        topic,
    )

    if pending_response is not None:
        return pending_response

    intent = route_message(request.message)
    route_source = "regex"

    if intent == Intent.UNKNOWN:
        intent = classify_unknown(request.message, client)
        route_source = "nano"

    if intent == Intent.CLARIFY and topic is None:
        return {
            "response": (
                "I can help you understand and access Australian healthcare services. "
                "What do you need help with?"
            )
        }

    if intent == Intent.CLARIFY and topic is not None:
        intent = Intent.NAVIGATION
        route_source = "memory"

    if intent == Intent.SOCIAL:
        log_route(
            intent,
            route_source,
            False,
        )

        if any(
            char in request.message
            for char in "谢谢謝你好您好再见見拜拜"
        ):
            return {
                "response": (
                    "谢谢！如果你需要了解如何使用澳大利亚的医疗服务，我可以帮助你。"
                )
            }

        return {
            "response": (
                "You're welcome. I can help if you need to navigate "
                "Australian health services."
            )
        }

    if intent == Intent.OUT_OF_SCOPE:
        log_route(
            intent,
            route_source,
            False,
        )

        return {
            "response": (
                "I can only help with navigating Australian healthcare services."
            )
        }

    if intent == Intent.EMERGENCY:
        log_route(
            intent,
            route_source,
            False,
        )

        return {
            "response": (
                "If you are seriously unwell, in immediate danger, "
                "or believe this is an emergency, "
                "call Triple Zero (000) or go to an emergency department."
            )
        }

    if intent == Intent.CLINICAL:
        log_route(
            intent,
            route_source,
            False,
        )

        return {
            "response": (
                REFUSAL_MESSAGE
                + " If you are unwell, a GP can usually help you decide what care you need. "
                "If you believe it is an emergency, call Triple Zero (000)."
            )
        }

    if (
        intent == Intent.NAVIGATION
        and (
            is_transport_request(request.message)
            or topic == "transport"
        )
    ):
        return handle_transport_request(
            request=request,
            intent=intent,
            topic=topic,
            client=client,
        )

    if (
        intent == Intent.NAVIGATION
        and is_service_finder_request(request.message)
    ):
        log_route(
            intent,
            "service_finder",
            False,
        )

        answer = (
            "You can use the Healthdirect Service Finder to find healthcare services near you.\n\n"
            "1. Enter the type of service you need, such as GP, pharmacy, hospital or urgent care.\n"
            "2. Enter your suburb or postcode.\n"
            "3. Select Search.\n"
            "4. Use Filters to narrow the results, for example by opening hours, fees or appointment options."
        )

        memory.add_turn(
            request.session_id,
            request.message,
            answer,
            topic=topic,
        )

        return {
            "response": answer,
            "link": "https://www.healthdirect.gov.au/australian-health-services",
            "images": [
                "/static/guides/healthdirect_search_en.png",
                "/static/guides/healthdirect_filter_en.png",
            ],
        }

    log_route(
        intent,
        route_source,
        True,
    )

    context = memory.get_context(
        request.session_id,
        topic=topic,
    )

    if context:
        model_input = (
            "Relevant conversation context:\n"
            f"{context}\n\n"
            "Current user message:\n"
            f"{request.message}"
        )
    else:
        model_input = request.message

    llm_start = time.perf_counter()

    response = client.responses.create(
        model="gpt-5-mini",
        instructions=SYSTEM_INSTRUCTION,
        input=model_input,
    )

    llm_latency_ms = round(
        (time.perf_counter() - llm_start) * 1000
    )

    log_llm(
        operation="navigation_response",
        model="gpt-5-mini",
        latency_ms=llm_latency_ms,
        usage=response.usage,
    )

    answer = response.output_text

    memory.add_turn(
        request.session_id,
        request.message,
        answer,
        topic=topic,
    )

    return {"response": answer}


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    topic = resolve_topic(
        request.session_id,
        request.message,
    )

    intent = route_message(request.message)
    route_source = "regex"

    if intent == Intent.UNKNOWN:
        intent = classify_unknown(request.message, client)
        route_source = "nano"

    if intent == Intent.CLARIFY and topic is not None:
        intent = Intent.NAVIGATION
        route_source = "memory"

    if intent != Intent.NAVIGATION:
        return chat(request)

    if (
        is_transport_request(request.message)
        or topic == "transport"
        or is_service_finder_request(request.message)
    ):
        return chat(request)

    log_route(
        intent,
        route_source,
        True,
    )

    context = memory.get_context(
        request.session_id,
        topic=topic,
    )

    if context:
        model_input = (
            "Relevant conversation context:\n"
            f"{context}\n\n"
            "Current user message:\n"
            f"{request.message}"
        )
    else:
        model_input = request.message

        def generate():
            full_response = ""
            stream_start = time.perf_counter()
            ttft_ms = None

            try:
                with client.responses.stream(
                    model="gpt-5-mini",
                    instructions=SYSTEM_INSTRUCTION,
                    input=model_input,
                ) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta":
                            if ttft_ms is None:
                                ttft_ms = round(
                                    (time.perf_counter() - stream_start) * 1000
                                )

                            full_response += event.delta
                            yield event.delta

                    final_response = stream.get_final_response()

                stream_latency_ms = round(
                    (time.perf_counter() - stream_start) * 1000
                )

                log_stream(
                    operation="navigation_response",
                    model="gpt-5-mini",
                    ttft_ms=ttft_ms,
                    stream_latency_ms=stream_latency_ms,
                )
                
                log_llm(
                    operation="navigation_response_stream",
                    model="gpt-5-mini",
                    latency_ms=stream_latency_ms,
                    usage=final_response.usage,
                )

                memory.add_turn(
                    request.session_id,
                    request.message,
                    full_response,
                    topic=topic,
                )

            except Exception:
                stream_latency_ms = round(
                    (time.perf_counter() - stream_start) * 1000
                )

                log_stream(
                    operation="navigation_response",
                    model="gpt-5-mini",
                    ttft_ms=ttft_ms,
                    stream_latency_ms=stream_latency_ms,
                    status="error",
                )

                raise

        return StreamingResponse(
            generate(),
            media_type="text/plain",
        )