from fastapi import FastAPI
from app.claim_model import get_claim_approval

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Health Claim Approval API is live!"}

@app.get("/predict")
def predict(query: str):
    decision = get_claim_approval(query)
    return {"decision": decision}
