from pydantic import BaseModel
from fastapi import APIRouter


class URLCheckRequest(BaseModel):
    url: str
    model: str


class URLCheckResponse(BaseModel):
    pred_prob: float
    is_malicious: bool


models = [
    {"name": "Model-1", "description": "First model."},
    {"name": "Model-2", "description": "Second model."}
]

router = APIRouter()


@router.get("/models")
async def get_models() -> list[dict[str, str]]:
    return models


@router.post("/predict", response_model=URLCheckResponse)
async def predict(request: URLCheckRequest) -> URLCheckResponse:
    # TODO: implement predict function
    return URLCheckResponse(pred_prob=0.5, is_malicious=True)


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
