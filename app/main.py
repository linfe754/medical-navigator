import os

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
from fastapi.responses import FileResponse

load_dotenv()

app = FastAPI(title="Medical Navigator")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REFUSAL_MESSAGE = (
    "Medical Navigator does not provide clinical advice, diagnosis, treatment "
    "recommendations, medication advice, or interpretation of symptoms or test results. "
    "I can help explain Australian health services, how to access care, referral pathways, "
    "appointments, costs, eligibility, and other non-clinical healthcare system information."
)

SYSTEM_INSTRUCTION = """
You are Medical Navigator, a non-clinical Australian healthcare system navigation assistant.

You may help users understand:
- Australian healthcare services
- Medicare and healthcare access
- referrals and appointment pathways
- public and private healthcare processes
- eligibility and administrative requirements
- healthcare costs and billing concepts
- how to locate appropriate healthcare services
- general explanations of how the healthcare system works

You must not provide:
- diagnosis or differential diagnosis
- assessment of symptoms
- treatment recommendations
- medication recommendations, dosing, changes, or comparisons
- interpretation of pathology, imaging, laboratory, or other clinical results
- prognosis
- personalised clinical advice
- decisions about whether a treatment is medically appropriate

If a request enters any of these clinical areas, do not attempt to answer the clinical part.
Return the fixed refusal message exactly as provided below:

""" + REFUSAL_MESSAGE


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return FileResponse("app/static/index.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat")
def chat(request: ChatRequest):
    response = client.responses.create(
        model="gpt-5-mini",
        instructions=SYSTEM_INSTRUCTION,
        input=request.message,
    )

    return {"response": response.output_text}
