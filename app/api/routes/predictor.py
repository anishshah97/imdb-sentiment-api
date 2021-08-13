from typing import Any

import joblib
from core.config import MODEL_LOADER
from core.errors import PredictException
from core.model_loaders import load_model_loaders
from fastapi import APIRouter, HTTPException
from loguru import logger
from models.prediction import (BERTSentimentRequest, BERTSentimentResponse,
                               HealthResponse)
from services.predict import BERTModelHandler as model

# from models.prediction import MachineLearningResponse
# from services.predict import MachineLearningModelHandlerScore as model

router = APIRouter()

# TODO: Build a better way to choose and switch between prediction functions
# def get_prediction(data_input): return MachineLearningResponse(
#     model.predict(
#         data_input, load_wrapper=load_model_loaders()[MODEL_LOADER], method="predict")
# )


@router.get("/predict", response_model=BERTSentimentResponse, name="predict:get-data")
# async def predict(request: BERTSentimentRequest):
async def predict(text: str):
    # text = request.text
    # if not request:
    if not text:
        raise HTTPException(
            status_code=404, detail=f"'data_input' argument invalid!")
    try:
        predictions = model.predict([text])
        sentiment_response = BERTSentimentResponse(
            text=text,
            sentiment=predictions["sentiment"],
            confidences=predictions["confidences"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception: {e}")

    return sentiment_response


@router.get(
    "/health", response_model=HealthResponse, name="health:get-data",
)
async def health():
    is_health = False
    try:
        test_text = [
            'This was an awesome movie. I watch it twice my time watching this beautiful movie if I have known it was this good']
        print(test_text)
        model.predict(test_text)
        is_health = True
        return HealthResponse(status=is_health)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail="Unhealthy")
