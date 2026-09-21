import os

import httpx
from dotenv import load_dotenv


load_dotenv()

ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
PLACES_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
PLACES_DETAILS_URL = "https://places.googleapis.com/v1/places"

def compute_transit_route(origin: str, destination: str) -> dict:
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    if not api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not configured")

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "routes.duration,"
            "routes.distanceMeters,"
            "routes.legs.steps.navigationInstruction,"
            "routes.legs.steps.transitDetails,"
            "routes.legs.steps.distanceMeters,"
            "routes.legs.steps.staticDuration"
        ),
    }

    payload = {
        "origin": {
            "address": origin,
        },
        "destination": {
            "address": destination,
        },
        "travelMode": "TRANSIT",
        "computeAlternativeRoutes": False,
        "languageCode": "en-AU",
        "units": "METRIC",
    }

    response = httpx.post(
        ROUTES_URL,
        headers=headers,
        json=payload,
        timeout=15.0,
    )

    response.raise_for_status()

    return response.json()

def compute_walking_route(
    origin_latitude: float,
    origin_longitude: float,
    destination_latitude: float,
    destination_longitude: float,
) -> dict:
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    if not api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not configured")

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "routes.distanceMeters,"
            "routes.duration"
        ),
    }

    payload = {
        "origin": {
            "location": {
                "latLng": {
                    "latitude": origin_latitude,
                    "longitude": origin_longitude,
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": destination_latitude,
                    "longitude": destination_longitude,
                }
            }
        },
        "travelMode": "WALK",
        "languageCode": "en-AU",
        "units": "METRIC",
    }

    response = httpx.post(
        ROUTES_URL,
        headers=headers,
        json=payload,
        timeout=15.0,
    )

    response.raise_for_status()

    result = response.json()

    routes = result.get("routes", [])

    if not routes:
        return {
            "distance_m": None,
            "duration_seconds": None,
        }

    route = routes[0]

    duration = route.get("duration", "0s")
    duration_seconds = int(float(duration.removesuffix("s")))

    return {
        "distance_m": route.get("distanceMeters"),
        "duration_seconds": duration_seconds,
    }
    
def get_place_entrances(place_id: str) -> dict:
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    if not api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not configured")

    headers = {
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "id,"
            "displayName,"
            "formattedAddress,"
            "location,"
            "entrances"
        ),
    }

    response = httpx.get(
        f"{PLACES_DETAILS_URL}/{place_id}",
        headers=headers,
        timeout=15.0,
    )

    response.raise_for_status()

    return response.json()


def find_nearest_transit(
    latitude: float,
    longitude: float,
    transit_type: str,
    radius: float = 3000,
) -> dict:
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    if not api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not configured")

    place_types = {
        "tram": "tram_stop",
        "bus": "bus_stop",
        "train": "train_station",
    }

    place_type = place_types.get(transit_type)

    if not place_type:
        raise ValueError("transit_type must be tram, bus, or train")

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "places.id,"
            "places.displayName,"
            "places.primaryType,"
            "places.location,"
            "places.transitStation"
        ),
    }

    payload = {
        "includedTypes": [place_type],
        "maxResultCount": 5,
        "rankPreference": "DISTANCE",
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": latitude,
                    "longitude": longitude,
                },
                "radius": radius,
            }
        },
    }

    response = httpx.post(
        PLACES_NEARBY_URL,
        headers=headers,
        json=payload,
        timeout=15.0,
    )

    response.raise_for_status()

    return response.json()

def parse_nearest_transit(result: dict, transit_type: str) -> dict | None:
    vehicle_types = {
        "tram": "TRAM",
        "bus": "BUS",
        "train": "HEAVY_RAIL",
    }

    vehicle_type = vehicle_types.get(transit_type)

    if not vehicle_type:
        raise ValueError("transit_type must be tram, bus, or train")

    for place in result.get("places", []):
        lines = []

        for agency in place.get("transitStation", {}).get("agencies", []):
            for line in agency.get("lines", []):
                if line.get("vehicleType") != vehicle_type:
                    continue

                short_name = line.get("shortDisplayName", {}).get("text")
                full_name = line.get("displayName", {}).get("text")

                lines.append({
                    "name": short_name or full_name,
                    "description": full_name,
                })

        if lines:
            return {
                "name": place.get("displayName", {}).get("text"),
                "location": place.get("location"),
                "lines": lines,
            }

    return None

def get_public_transport_access(
    latitude: float,
    longitude: float,
    facility_address: str,
) -> dict:
    access = {}

    for transit_type in ["tram", "bus", "train"]:
        result = find_nearest_transit(
            latitude,
            longitude,
            transit_type,
        )

        parsed = parse_nearest_transit(
            result,
            transit_type,
        )

        if not parsed:
            access[transit_type] = None
            continue

        stop_location = parsed.get("location")

        if not stop_location:
            access[transit_type] = parsed
            continue

        walking = compute_walking_route_from_address(
            facility_address,
            stop_location["latitude"],
            stop_location["longitude"],
        )

        parsed["walking_distance_m"] = walking["distance_m"]
        parsed["walking_duration_seconds"] = walking["duration_seconds"]

        access[transit_type] = parsed

    return access



def resolve_healthcare_facility(query: str) -> list[dict]:
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    if not api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not configured")

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "places.id,"
            "places.displayName,"
            "places.formattedAddress,"
            "places.location,"
            "places.types"
        ),
    }

    payload = {
        "textQuery": query,
        "languageCode": "en-AU",
        "regionCode": "AU",
        "pageSize": 5,
    }

    response = httpx.post(
        PLACES_TEXT_SEARCH_URL,
        headers=headers,
        json=payload,
        timeout=15.0,
    )

    response.raise_for_status()

    result = response.json()

    facilities = []

    for place in result.get("places", []):
        facilities.append(
            {
                "place_id": place.get("id"),
                "name": place.get(
                    "displayName", {}
                ).get("text"),
                "address": place.get("formattedAddress"),
                "location": place.get("location"),
                "types": place.get("types", []),
            }
        )

    return facilities

def get_facility_transport(query: str) -> dict:
    facilities = resolve_healthcare_facility(query)

    if not facilities:
        return {
            "status": "not_found",
            "query": query,
            "facility": None,
            "candidates": [],
            "public_transport": None,
        }

    if len(facilities) > 1:
        return {
            "status": "ambiguous",
            "query": query,
            "facility": None,
            "candidates": facilities,
            "public_transport": None,
        }

    facility = facilities[0]

    location = facility.get("location")

    if not location:
        return {
            "status": "error",
            "query": query,
            "facility": facility,
            "candidates": [],
            "public_transport": None,
        }

    public_transport = get_public_transport_access(
        location["latitude"],
        location["longitude"],
        facility["address"],
    )

    return {
        "status": "resolved",
        "query": query,
        "facility": {
            "place_id": facility["place_id"],
            "name": facility["name"],
            "address": facility["address"],
            "location": location,
        },
        "candidates": [],
        "public_transport": public_transport,
    }
    
def compute_walking_route_from_address(
    origin_address: str,
    destination_latitude: float,
    destination_longitude: float,
) -> dict:
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    if not api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not configured")

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "routes.distanceMeters,"
            "routes.duration"
        ),
    }

    payload = {
        "origin": {
            "address": origin_address,
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": destination_latitude,
                    "longitude": destination_longitude,
                }
            }
        },
        "travelMode": "WALK",
        "languageCode": "en-AU",
        "units": "METRIC",
    }

    response = httpx.post(
        ROUTES_URL,
        headers=headers,
        json=payload,
        timeout=15.0,
    )

    response.raise_for_status()

    result = response.json()
    routes = result.get("routes", [])

    if not routes:
        return {
            "distance_m": None,
            "duration_seconds": None,
        }

    route = routes[0]
    duration = route.get("duration", "0s")

    return {
        "distance_m": route.get("distanceMeters"),
        "duration_seconds": int(float(duration.removesuffix("s"))),
    }