from typing import Optional

from pydantic import BaseModel


class MachineLearningResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: bool
