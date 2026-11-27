import os
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT_DIR / "models" / "v5_benchmark.cbm")))
MODEL_FEATURES = [
    "Age", "Driving_License", "Region_Code", "Previously_Insured",
    "Annual_Premium", "Policy_Sales_Channel", "Vintage",
    "Vehicle_Age_enc", "Vehicle_Damage_enc", "Gender_enc",
]

app = FastAPI(title="Insurance Prediction API")
model = None


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Age: int = Field(ge=18, le=100)
    Previously_Insured: Literal[0, 1]
    Annual_Premium: int = Field(ge=0)
    Vintage: int = Field(ge=0)
    Region_Code: int = Field(ge=0)
    Policy_Sales_Channel: int = Field(ge=0)
    Vehicle_Age: Literal["< 1 Year", "1-2 Year", "> 2 Years"]
    Vehicle_Damage: Literal["Yes", "No"]
    Gender: Literal["Male", "Female"]
    Driving_License: Literal[0, 1]


class PredictionResponse(BaseModel):
    probability: float = Field(ge=0, le=1)


def _load_model():
    global model

    if model is not None:
        return model
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Модель не найдена. Поместите совместимую модель v5 в models/v5_benchmark.cbm.",
        )

    from catboost import CatBoostClassifier

    loaded_model = CatBoostClassifier()
    loaded_model.load_model(str(MODEL_PATH))
    if loaded_model.feature_names_ != MODEL_FEATURES:
        raise HTTPException(
            status_code=503,
            detail="Модель несовместима с API: ожидаются признаки benchmark v5.",
        )

    model = loaded_model
    return model


def _build_model_input(request: PredictionRequest) -> pd.DataFrame:
    values = request.model_dump()
    vehicle_age_encoding = {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}

    row = {
        "Age": values["Age"],
        "Driving_License": values["Driving_License"],
        "Region_Code": str(values["Region_Code"]),
        "Previously_Insured": values["Previously_Insured"],
        "Annual_Premium": values["Annual_Premium"],
        "Policy_Sales_Channel": str(values["Policy_Sales_Channel"]),
        "Vintage": values["Vintage"],
        "Vehicle_Age_enc": vehicle_age_encoding[values["Vehicle_Age"]],
        "Vehicle_Damage_enc": int(values["Vehicle_Damage"] == "Yes"),
        "Gender_enc": int(values["Gender"] == "Male"),
    }
    return pd.DataFrame([row], columns=MODEL_FEATURES)


@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_file_available": MODEL_PATH.is_file(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    prediction_model = _load_model()
    probabilities = prediction_model.predict_proba(_build_model_input(request))
    return PredictionResponse(probability=float(probabilities[0][1]))