from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.pose_ml_service import predict_from_video_file

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