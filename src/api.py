"""FastAPI service. Run from the repo root:  uvicorn src.api:app"""
import json
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from src.config import RESULTS_DIR, SERVE_MODEL_DIR
from src.data import combine_text

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load everything once at startup."""
    from src.embeddings import load_encoder

    STATE["encoder"] = load_encoder()
    STATE["clf"] = joblib.load(SERVE_MODEL_DIR / "classifier.joblib")
    STATE["labels"] = json.loads((SERVE_MODEL_DIR / "labels.json").read_text(encoding="utf-8"))
    if [str(c) for c in STATE["clf"].classes_] != STATE["labels"]:
        raise RuntimeError("labels.json does not match the classifier's class order")
    STATE["threshold"] = json.loads(
        (RESULTS_DIR / "abstention_selection.json").read_text(encoding="utf-8"))["threshold"]
    yield
    STATE.clear()


app = FastAPI(title="News topic classifier", lifespan=lifespan)


class PredictRequest(BaseModel):
    headline: str = Field(max_length=1000)
    short_description: str | None = Field(default=None, max_length=5000)

    @field_validator("headline")
    @classmethod
    def headline_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("headline must not be blank")
        return v


class PredictResponse(BaseModel):
    category: str | None   # null when the model abstains
    confidence: float      # highest class probability; a practical score, not a guarantee
    abstain: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "threshold": STATE["threshold"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    from src.embeddings import encode_texts

    text = combine_text(req.headline, req.short_description)  # same function as training
    X = encode_texts(STATE["encoder"], [text], show_progress=False)
    proba = STATE["clf"].predict_proba(X)[0]
    top = int(proba.argmax())
    confidence = float(proba[top])
    abstain = confidence < STATE["threshold"]
    return PredictResponse(
        category=None if abstain else STATE["labels"][top],
        confidence=round(confidence, 4),
        abstain=abstain,
    )