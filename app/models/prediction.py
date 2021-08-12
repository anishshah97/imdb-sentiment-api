from typing import Dict, Optional

from pydantic import BaseModel


class MachineLearningResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: bool


class BERTSentimentRequest(BaseModel):
    text: str


class BERTSentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidences: Dict[str, float]
