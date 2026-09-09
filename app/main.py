from fastapi import FastAPI

app = FastAPI(title="Medical Navigator")

@app.get("/")
def root():
    return {"message": "Medical Navigator is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}
