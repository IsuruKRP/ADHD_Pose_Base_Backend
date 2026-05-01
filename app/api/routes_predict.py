from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.pose_ml_service import predict_from_video_file, predict_from_features

router = APIRouter()

@router.post("/predict-test")
async def predict_test(
    video: UploadFile = File(...),
    rounds: int = Form(...),
):
    try:
        video_bytes = await video.read()
        result = predict_from_video_file(video_bytes)
        return {
            "status": "success",
            "message": "Prediction successful (Test Mode)",
            "result": result
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}


@router.post("/predict")
async def predict_video(
    video: UploadFile = File(...),
    rounds: int = Form(...),
    age: int = Form(None),
    gender: str = Form(None),
    group: str = Form(None),
):
    try:
        video_bytes = await video.read()

        result = predict_from_video_file(video_bytes)

        return {
            "age": age,
            "gender": gender,
            "rounds": rounds,
            "subtype": result["subtype"],
            "adhd_score": result["adhd_score"],
            "adhd_probability": result["adhd_probability"],
            "derived_features": result["derived_features"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# NEW: Fast prediction from pre-computed frontend features
# ─────────────────────────────────────────────────────────────

class FeaturesInput(BaseModel):
    fidget_score: float
    sway_x: float
    sway_y: float
    mean_velocity: float
    max_velocity: float
    std_velocity: float
    mean_abs_displacement: float
    stability_score: float
    rounds: Optional[int] = None
    age: Optional[int] = None
    gender: Optional[str] = None


@router.post("/predict-features")
async def predict_features_endpoint(data: FeaturesInput):
    """
    Fast inference endpoint.
    Accepts 8 pre-computed pose features from the browser (MediaPipe WASM).
    Runs only StandardScaler + Random Forest — no video, no OpenCV, no MediaPipe on server.
    Returns result in < 100ms.
    """
    try:
        features = {
            "fidget_score": data.fidget_score,
            "sway_x": data.sway_x,
            "sway_y": data.sway_y,
            "mean_velocity": data.mean_velocity,
            "max_velocity": data.max_velocity,
            "std_velocity": data.std_velocity,
            "mean_abs_displacement": data.mean_abs_displacement,
            "stability_score": data.stability_score,
        }
        result = predict_from_features(features)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-test")
async def predict_test(
    video: UploadFile = File(...),
    rounds: int = Form(...),
):
    try:
        video_bytes = await video.read()
        result = predict_from_video_file(video_bytes)
        return {
            "status": "success",
            "message": "Prediction successful (Test Mode)",
            "result": result
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}


@router.post("/predict")
async def predict_video(
    video: UploadFile = File(...),
    rounds: int = Form(...),
    age: int = Form(None),
    gender: str = Form(None),
    group: str = Form(None),
):
    try:
        video_bytes = await video.read()

        result = predict_from_video_file(video_bytes)

        return {
            "age": age,
            "gender": gender,
            "rounds": rounds,
            "subtype": result["subtype"],
            "adhd_score": result["adhd_score"],
            "adhd_probability": result["adhd_probability"],
            "derived_features": result["derived_features"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))