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

class ScanRequest(BaseModel):
    prompt: str = "person loitering near entrance"
    threshold: float = 0.5
    n_results: int = 10

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

@app.post("/agent/scan")
def run_scan(request: ScanRequest):
    from src.agents.anomaly import score_results
    results = search_videos(request.prompt, n_results=request.n_results, video=False)
    scored = score_results(results, clip_threshold=1.4)
    flagged = [s for s in scored if s.anomaly_score >= request.threshold]
    return {
        "flagged": len(flagged),
        "anomalies": [
            {
                "scene": a.scene,
                "frame_idx": a.frame_idx,
                "image_path": a.image_path,
                "score": a.anomaly_score,
                "detected_classes": list(set(a.detected_classes)),
                "reasons": a.reasons,
            }
            for a in flagged
        ]
    }