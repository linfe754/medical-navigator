import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    return OpenAI(api_key=api_key)
