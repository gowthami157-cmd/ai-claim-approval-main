import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def claim_format(row):
  return f"Patient: {row['PatientGender']}, {row['PatientAge']} years old, {row['PatientMaritalStatus']} and {row['PatientEmploymentStatus']}. " \
         f"Diagnosis: {row['DiagnosisCode']}, Procedure: {row['ProcedureCode']}. " \
         f"Provider Specialty: {row['ProviderSpecialty']}. " \
         f"Claim Type: {row['ClaimType']} via {row['ClaimSubmissionMethod']}. " \
         f"Claim Amount: ${row['ClaimAmount']}. Claim Status: {row['ClaimStatus']}."

df = pd.read_csv("data/health_claims.csv")
columns_needed = [
    "PatientAge", "PatientGender", "PatientIncome", "PatientMaritalStatus", "PatientEmploymentStatus",
    "ClaimAmount", "ClaimType", "ClaimSubmissionMethod", "DiagnosisCode", "ProcedureCode",
    "ProviderSpecialty", "ClaimStatus"
]

df = df[columns_needed]
df['ClaimText'] = df.apply(claim_format, axis = 1)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(df["ClaimText"].tolist())

d = embeddings.shape[1]  
index = faiss.IndexFlatL2(d)  
index.add(np.array(embeddings)) 

# Save the FAISS index
faiss.write_index(index, "models/claims_faiss.index")
print("✅ FAISS Index Successfully Built and Saved!")
