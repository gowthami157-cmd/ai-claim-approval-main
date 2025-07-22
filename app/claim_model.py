import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai

from app.config import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("models/claims_faiss.index")

def claim_format(row):
  return f"Patient: {row['PatientGender']}, {row['PatientAge']} years old, {row['PatientMaritalStatus']} and {row['PatientEmploymentStatus']}. " \
         f"Diagnosis: {row['DiagnosisCode']}, Procedure: {row['ProcedureCode']}. " \
         f"Provider Specialty: {row['ProviderSpecialty']}. " \
         f"Claim Type: {row['ClaimType']} via {row['ClaimSubmissionMethod']}. " \
         f"Claim Amount: ${row['ClaimAmount']}. Claim Status: {row['ClaimStatus']}."

DATA_PATH = "data/health_claims.csv" 
df = pd.read_csv(DATA_PATH)
columns_needed = [
    "PatientAge", "PatientGender", "PatientIncome", "PatientMaritalStatus", "PatientEmploymentStatus",
    "ClaimAmount", "ClaimType", "ClaimSubmissionMethod", "DiagnosisCode", "ProcedureCode",
    "ProviderSpecialty", "ClaimStatus"
]

df = df[columns_needed]
df['ClaimText'] = df.apply(claim_format, axis = 1)

def search_similar_claims(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    similar_claims = []
    for i in range(top_k):
        claim_text = f"Claim: {df.iloc[indices[0][i]]['ClaimText']} (Score: {distances[0][i]:.4f})"
        similar_claims.append(claim_text)

    return "\n".join(similar_claims)

def get_claim_approval(query):
    similar_claims = search_similar_claims(query, top_k=3)

    prompt = f"""
    You are an AI system that processes health insurance claims. 
    Your task is to decide whether a new claim should be **approved** or **denied**, based on past claim records.

    **New Claim:**
    {query}

    **Past Similar Claims Retrieved:**
    {similar_claims}

    **Decision Task:**
    - Compare the **new claim** with past similar claims.
    - If similar claims were approved, approve the new claim unless there are strong reasons not to.
    - If most similar claims were denied, deny the new claim unless there is an exception.
    - If past claims are mixed, analyze the **claim type, provider specialty, diagnosis, and procedure** before making a decision.
    - Clearly state **Approved** or **Denied**, followed by a short reasoning.

    **Output Format:**
    - Decision: (Approved/Denied)
    - Reason: (Explain based on retrieved claims)
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
