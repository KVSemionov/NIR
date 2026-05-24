from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.predict import ModelPredictor
from src.logger import setup_logger, log_api_request
from src.monitor import PredictionLogger

app = FastAPI(
    title="Student Performance Prediction API",
    description="API for predicting student academic performance",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = setup_logger("api")
prediction_logger = PredictionLogger()

MODEL_PATH = "models/model_v1.pkl"
predictor = None


class BatchPredictionInput(BaseModel):
    """Модель для пакетного предсказания."""

    inputs: List[Dict[str, Any]] = Field(..., description="Список входных данных")


class PredictionOutput(BaseModel):
    """Модель вывода предсказания."""
    
    prediction: float = Field(..., description="Предсказанная оценка")
    model_version: str = Field(..., description="Версия модели")
    input_data: Dict[str, Any] = Field(..., description="Входные данные")


class BatchPredictionOutput(BaseModel):
    """Модель вывода пакетного предсказания."""
    
    predictions: List[PredictionOutput] = Field(..., description="Список предсказаний")
    total: int = Field(..., description="Общее количество предсказаний")


class HealthResponse(BaseModel):
    """Модель ответа health check."""
    
    status: str
    model_loaded: bool
    model_version: Optional[str]


class ModelInfoResponse(BaseModel):
    """Модель информации о модели."""
    
    model_type: str
    model_version: str
    feature_names: List[str]
    metrics: Dict[str, float]


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске API."""
    global predictor
    logger.info("Starting API server")
    
    try:
        predictor = ModelPredictor(MODEL_PATH, logger)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware для логирования всех запросов."""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration_ms = (time.time() - start_time) * 1000
    log_api_request(logger, str(request.url.path), request.method, response.status_code, duration_ms)
    
    return response


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья API."""
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        model_version=predictor.model_version if predictor else None
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Получение информации о модели."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_type=predictor.trainer.model_type,
        model_version=predictor.model_version,
        feature_names=predictor.trainer.feature_names,
        metrics=predictor.trainer.metrics
    )


@app.get("/model/features")
async def get_feature_importance():
    """Получение важности признаков."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    importance = predictor.get_feature_importance()
    return {"feature_importance": importance}


@app.get("/model/input-example")
async def get_input_example():
    """Пример корректного тела запроса для /predict."""
    return {
        "Подразделение": "Институт информационных технологий и компьютерных наук",
        "Курс_по_порядку": 4,
        "history_mean": 4.2,
        "last_mark": 4.5,
        "EGE_Английский_язык": 0,
        "EGE_Информатика_и_ИКТ": 80,
        "EGE_История": 0,
        "EGE_Математика": 76,
        "EGE_Обществознание": 0,
        "EGE_Русский_язык": 85,
        "EGE_Физика": 0,
        "EGE_Химия": 0,
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: Dict[str, Any]):
    """
    Предсказание оценки для одного студента.
    
    Args:
        input_data: Входные данные студента
        
    Returns:
        Предсказанная оценка
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_dict = dict(input_data)
        prediction = predictor.predict(input_dict)
        
        prediction_logger.log(
            input_data=input_dict,
            prediction=prediction,
            model_version=predictor.model_version
        )
        
        return PredictionOutput(
            prediction=prediction,
            model_version=predictor.model_version,
            input_data=input_dict
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Пакетное предсказание оценок для нескольких студентов.
    
    Args:
        batch_input: Список входных данных студентов
        
    Returns:
        Список предсказаний
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        inputs_list = [dict(item) for item in batch_input.inputs]
        results = predictor.predict_batch(inputs_list)
        
        predictions = []
        for result in results:
            prediction_logger.log(
                input_data=result['input_data'],
                prediction=result['prediction'],
                model_version=predictor.model_version
            )
            
            predictions.append(PredictionOutput(
                prediction=result['prediction'],
                model_version=predictor.model_version,
                input_data=result['input_data']
            ))
        
        return BatchPredictionOutput(
            predictions=predictions,
            total=len(predictions)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/logs")
async def get_prediction_logs(n_last: int = 100):
    """
    Получение последних логов предсказаний.
    
    Args:
        n_last: Количество последних записей
        
    Returns:
        Список логов предсказаний
    """
    logs = prediction_logger.read_logs(n_last)
    return {"logs": logs, "total": len(logs)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
