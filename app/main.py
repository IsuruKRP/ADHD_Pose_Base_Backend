from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes_predict import router as predict_router

app = FastAPI(title="ADHD Care Portal API")

# CORS — allow all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "name": "ADHD Body Posture Inference Backend",
        "endpoints": [
            "/health",
            "/predict",
            "/predict-test"
        ]
    }

@app.get("/health")
async def health():
    return {"ok": True}

app.include_router(predict_router)
