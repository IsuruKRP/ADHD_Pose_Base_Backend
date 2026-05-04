from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_predict import router as predict_router

app = FastAPI(title="ADHD Body Posture Inference Service")

# The frontend should normally call the API gateway. Direct CORS is kept open
# for local service testing and isolated deployment checks.
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
        "service": "posture",
        "endpoints": [
            "/health",
            "/predict",
            "/predict-test",
            "/predict-features",
        ],
    }


@app.get("/health")
async def health():
    return {"ok": True, "service": "posture"}


app.include_router(predict_router)
