from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import clip
from pinecone import Pinecone

# 1. Initialize FastAPI
app = FastAPI()

# 2. Add CORS Middleware (MUST be here, before your endpoints)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
pc = Pinecone(api_key="pcsk_6UGTG1_NbStZZy8DVSCppTteVetowHiwKy5TcCiuxMg6x8VcgxC4MKxkioMLiKBGp21XbM") 
index = pc.Index("fashion-search")

# 4. Request Schema
class SearchRequest(BaseModel):
    query: str

# 5. Search Endpoint
@app.post("/search")
async def search(request: SearchRequest):
    # Convert text to 512-dim vector
    text = clip.tokenize([request.query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        query_vector = text_features.cpu().numpy().tolist()[0]
    
    # Query Pinecone
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)
    
    # Extract the data into a clean list of dictionaries for the frontend
    clean_results = []
    for match in results['matches']:
        clean_results.append({
            "id": match['id'],
            "score": match['score'],
            "metadata": match['metadata']
        })
        
    return {"results": clean_results}