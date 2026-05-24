import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "ml_project", log_dir: str = "logs") -> logging.Logger:
    """
    Настройка логгера с записью в файл и консоль.
    
    Args:
        name: Имя логгера
        log_dir: Директория для хранения логов
        
    Returns:
        Настроенный логгер
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(
        log_dir_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_prediction(logger: logging.Logger, input_data: dict, prediction: float, model_version: str = "v1"):
    """
    Логирование предсказания.
    
    Args:
        logger: Логгер
        input_data: Входные данные
        prediction: Предсказание модели
        model_version: Версия модели
    """
    logger.info(f"Prediction | Model: {model_version} | Input: {input_data} | Prediction: {prediction}")


def log_model_metrics(logger: logging.Logger, metrics: dict, model_version: str = "v1"):
    """
    Логирование метрик модели.
    
    Args:
        logger: Логгер
        metrics: Словарь с метриками
        model_version: Версия модели
    """
    logger.info(f"Model Metrics | Model: {model_version} | Metrics: {metrics}")


def log_api_request(logger: logging.Logger, endpoint: str, method: str, status_code: int, duration_ms: float):
    """
    Логирование API запроса.
    
    Args:
        logger: Логгер
        endpoint: Эндпоинт
        method: HTTP метод
        status_code: Код ответа
        duration_ms: Время выполнения в мс
    """
    logger.info(f"API Request | {method} {endpoint} | Status: {status_code} | Duration: {duration_ms:.2f}ms")
