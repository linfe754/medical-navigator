import json
import logging
from contextvars import ContextVar

logger = logging.getLogger("medical_navigator")

request_id_ctx: ContextVar[str] = ContextVar(
    "request_id",
    default="unknown",
)


def log_route(intent, route_source: str, main_model: bool):
    logger.info(
        json.dumps(
            {
                "event": "route",
                "request_id": request_id_ctx.get(),
                "intent": intent.value,
                "route": route_source,
                "main_model": main_model,
            }
        )
    )


def log_tool(tool: str, latency_ms: int, status: str):
    logger.info(
        json.dumps(
            {
                "event": "tool_usage",
                "request_id": request_id_ctx.get(),
                "tool": tool,
                "latency_ms": latency_ms,
                "status": status,
            }
        )
    )
    
def log_llm(
    operation: str,
    model: str,
    latency_ms: int,
    usage,
    status: str = "success",
):
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)

    logger.info(
        json.dumps(
            {
                "event": "llm_usage",
                "request_id": request_id_ctx.get(),
                "operation": operation,
                "model": model,
                "status": status,
                "input_tokens": getattr(usage, "input_tokens", None),
                "cached_tokens": getattr(
                    input_details,
                    "cached_tokens",
                    None,
                ),
                "output_tokens": getattr(usage, "output_tokens", None),
                "reasoning_tokens": getattr(
                    output_details,
                    "reasoning_tokens",
                    None,
                ),
                "total_tokens": getattr(usage, "total_tokens", None),
                "llm_latency_ms": latency_ms,
            }
        )
    )
    
       