# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from query import search_videos

app = FastAPI(
    title="VideoLens",
    description="Multimodal Video Understanding and Semantic Search",
    version="0.1.0"
)

class SearchRequest(BaseModel):
    query: str
    n_results: int = 5
    video: bool = False
@app.post("/search")
def search(request: SearchRequest):
    try:
        results = search_videos(request.query, request.n_results, request.video)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "VideoLens API"}

@app.get("/health")
def health():
    return {"status": "healthy"}