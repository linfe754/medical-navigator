import time

from app.memory import memory
from app.observability.telemetry import log_route, log_tool
from app.router import (
    extract_transport_destination,
    extract_transport_origin,
)
from app.tools.transport import (
    compute_transit_route,
    get_facility_transport,
)


pending_facilities: dict[str, list[dict]] = {}


def format_transport_response(result: dict) -> str:
    facility = result["facility"]
    transport = result["public_transport"]

    lines = [
        f"{facility['name']}",
        f"{facility['address']}",
        "",
        "Public transport access:",
    ]

    for mode in ["tram", "bus", "train"]:
        item = transport.get(mode)

        if not item:
            continue

        services = ", ".join(
            line["name"]
            for line in item["lines"]
            if line.get("name")
        )

        minutes = round(item["walking_duration_seconds"] / 60)

        lines.append(
            f"{mode.title()}: {item['name']} — {services}. "
            f"About {item['walking_distance_m']} m walk (~{minutes} min)."
        )

    return "\n".join(lines)


def format_transit_journey(
    route: dict,
    origin: str,
    facility: dict,
) -> str:
    routes = route.get("routes", [])

    if not routes:
        return "I couldn't find a public transport route for that journey."

    selected_route = routes[0]
    steps = selected_route["legs"][0]["steps"]

    lines = [
        f"Public transport from {origin} to {facility['name']}:",
        "",
    ]

    number = 1

    for step in steps:
        transit = step.get("transitDetails")

        if not transit:
            continue

        transit_line = transit["transitLine"]
        stops = transit["stopDetails"]

        vehicle = transit_line["vehicle"]["name"]["text"]
        line_name = transit_line.get("nameShort") or transit_line.get("name")

        departure = stops["departureStop"]["name"]
        arrival = stops["arrivalStop"]["name"]

        lines.append(f"{number}. {vehicle} — {line_name}")
        lines.append(f"   {departure} → {arrival}")
        lines.append("")

        number += 1

    duration_seconds = int(selected_route["duration"].rstrip("s"))
    duration_minutes = round(duration_seconds / 60)

    lines.append(
        f"Typical journey time: about {duration_minutes} minutes"
    )

    return "\n".join(lines)


def handle_pending_facility_selection(
    request,
    topic: str | None,
):
    pending = pending_facilities.get(request.session_id)

    if not pending:
        return None

    if request.message.strip() not in {"1", "2", "3"}:
        return None

    choice = int(request.message.strip()) - 1

    if choice >= len(pending):
        return None

    facility = pending[choice]
    pending_facilities.pop(request.session_id, None)

    tool_start = time.perf_counter()
    tool_status = "error"

    try:
        result = get_facility_transport(facility["name"])
        tool_status = result.get("status", "success")
    finally:
        log_tool(
            "get_facility_transport",
            round((time.perf_counter() - tool_start) * 1000),
            tool_status,
        )

    if result["status"] != "resolved":
        return None

    answer = format_transport_response(result)

    memory.set_selected_service(
        request.session_id,
        result["facility"]["name"],
        result["facility"]["address"],
    )

    memory.add_turn(
        request.session_id,
        request.message,
        answer,
        topic=topic,
    )

    return {"response": answer}


def handle_transport_request(
    request,
    intent,
    topic: str | None,
    client,
):
    destination = extract_transport_destination(
        request.message,
        client,
    )

    origin = extract_transport_origin(
        request.message,
        client,
    )

    selected_service = memory.get_session(
        request.session_id
    ).state.selected_service

    if not destination and selected_service:
        destination = selected_service.name

    if not destination:
        return {
            "response": (
                "Which healthcare facility do you want to travel to?"
            )
        }

    tool_start = time.perf_counter()
    tool_status = "error"

    try:
        result = get_facility_transport(destination)
        tool_status = result.get("status", "success")
    finally:
        log_tool(
            "get_facility_transport",
            round((time.perf_counter() - tool_start) * 1000),
            tool_status,
        )

    if result["status"] == "not_found":
        return {
            "response": (
                "I couldn't find that healthcare facility. "
                "Please give me the facility name or suburb."
            )
        }

    if result["status"] == "ambiguous":
        candidates = []
        seen_addresses = set()

        for item in result["candidates"]:
            address = item.get("address")

            if address in seen_addresses:
                continue

            seen_addresses.add(address)
            candidates.append(item)

            if len(candidates) == 3:
                break

        pending_facilities[request.session_id] = candidates

        options = "\n".join(
            f"{i + 1}. {item['name']} — {item['address']}"
            for i, item in enumerate(candidates)
        )

        return {
            "response": (
                "I found several possible healthcare facilities. "
                f"Which one do you mean?\n\n{options}"
            )
        }

    memory.set_selected_service(
        request.session_id,
        result["facility"]["name"],
        result["facility"]["address"],
    )

    if origin:
        memory.set_location(
            request.session_id,
            origin,
        )

        tool_start = time.perf_counter()
        tool_status = "error"

        try:
            route = compute_transit_route(
                origin,
                result["facility"]["address"],
            )
            tool_status = "success"
        finally:
            log_tool(
                "compute_transit_route",
                round((time.perf_counter() - tool_start) * 1000),
                tool_status,
            )

        log_route(
            intent,
            "transport_route_tool",
            False,
        )

        answer = format_transit_journey(
            route,
            origin,
            result["facility"],
        )

        memory.add_turn(
            request.session_id,
            request.message,
            answer,
            topic=topic,
        )

        return {"response": answer}

    log_route(
        intent,
        "transport_tool",
        False,
    )

    answer = format_transport_response(result)

    memory.add_turn(
        request.session_id,
        request.message,
        answer,
        topic=topic,
    )

    return {"response": answer}