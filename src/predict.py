import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from typing import Dict, List, Union
from .logger import setup_logger, log_prediction
from .train import ModelTrainer


class ModelPredictor:
    """Класс для предсказания с помощью обученной модели."""
    
    def __init__(self, model_path: str, logger=None):
        self.model_path = model_path
        self.logger = logger or setup_logger("predict")
        self.trainer = ModelTrainer(logger=logger)
        self.trainer.load_model(model_path)
        self.model_version = self._extract_version()
    
    def _extract_version(self) -> str:
        """Извлечение версии модели из пути."""
        if "v1" in self.model_path:
            return "v1"
        elif "v2" in self.model_path:
            return "v2"
        return "unknown"
    
    def prepare_input(
        self, 
        input_data: Union[Dict, List[Dict], pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Подготовка входных данных для предсказания.
        
        Args:
            input_data: Входные данные (словарь, список словарей или DataFrame)
            
        Returns:
            DataFrame с признаками
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError("Input data must be dict, list of dicts, or DataFrame")
        
        self.logger.info(f"Prepared input data with {len(df)} samples")
        
        return df
    
    def validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Проверка и выравнивание признаков с обученной моделью.
        
        Args:
            df: DataFrame с признаками
            
        Returns:
            DataFrame с выровненными признаками
        """
        required_features = self.trainer.feature_names
        available_features = df.columns.tolist()
        
        missing_features = set(required_features) - set(available_features)
        extra_features = set(available_features) - set(required_features)
        
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}. Filling with 0.")
            for feature in missing_features:
                df[feature] = "" if feature == "Подразделение" else 0
        
        if extra_features:
            self.logger.info(f"Extra features will be ignored: {extra_features}")
        
        df = df[required_features].copy()

        # В pipeline есть OneHotEncoder для строкового признака и числовые признаки.
        # Если заполнить категориальный признак числом 0, sklearn получает смесь int/str
        # и падает с ошибкой "'<' not supported between instances of 'int' and 'str'".
        if "Подразделение" in df.columns:
            df["Подразделение"] = df["Подразделение"].fillna("").astype(str)

        for feature in required_features:
            if feature != "Подразделение":
                df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0)
        
        return df
    
    def predict(
        self, 
        input_data: Union[Dict, List[Dict], pd.DataFrame],
        return_proba: bool = False
    ) -> Union[float, List[float], np.ndarray]:
        """
        Предсказание оценки успеваемости.
        
        Args:
            input_data: Входные данные
            return_proba: Возвращать ли вероятности (для классификации)
            
        Returns:
            Предсказанные оценки
        """
        self.logger.info("Making prediction")
        
        df = self.prepare_input(input_data)
        df = self.validate_features(df)
        
        predictions = self.trainer.model.predict(df)
        
        if len(predictions) == 1:
            prediction = float(predictions[0])
            log_prediction(self.logger, input_data if isinstance(input_data, dict) else input_data[0], prediction, self.model_version)
            return prediction
        else:
            predictions_list = predictions.tolist()
            for i, pred in enumerate(predictions_list):
                log_prediction(self.logger, input_data[i] if isinstance(input_data, list) else input_data.iloc[i].to_dict(), pred, self.model_version)
            return predictions_list
    
    def predict_batch(
        self, 
        input_data: Union[List[Dict], pd.DataFrame]
    ) -> List[Dict]:
        """
        Пакетное предсказание с дополнительной информацией.
        
        Args:
            input_data: Входные данные
            
        Returns:
            Список словарей с предсказаниями и метаданными
        """
        self.logger.info(f"Making batch prediction for {len(input_data)} samples")
        
        df = self.prepare_input(input_data)
        df = self.validate_features(df)
        
        predictions = self.trainer.model.predict(df)
        
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'prediction': float(pred),
                'model_version': self.model_version,
                'input_index': i
            }
            
            if isinstance(input_data, pd.DataFrame):
                result['input_data'] = input_data.iloc[i].to_dict()
            else:
                result['input_data'] = input_data[i]
            
            log_prediction(self.logger, result['input_data'], pred, self.model_version)
            results.append(result)
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Получение важности признаков.
        
        Returns:
            Словарь с важностью признаков
        """
        model = self.trainer.model
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            model = model.named_steps["model"]

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = self.trainer.feature_names

            if len(importance) != len(feature_names):
                self.logger.warning(
                    "Feature importance length does not match original feature names after preprocessing"
                )
                return {}
            
            feature_importance = dict(zip(feature_names, importance))
            
            sorted_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            
            self.logger.info(f"Feature importance: {sorted_importance}")
            
            return sorted_importance
        else:
            self.logger.warning("Model does not support feature importance")
            return {}


def predict_single(
    model_path: str,
    input_data: Dict
) -> float:
    """
    Быстрое предсказание для одного примера.
    
    Args:
        model_path: Путь к модели
        input_data: Входные данные
        
    Returns:
        Предсказанная оценка
    """
    predictor = ModelPredictor(model_path)
    return predictor.predict(input_data)


def predict_batch(
    model_path: str,
    input_data: Union[List[Dict], pd.DataFrame]
) -> List[Dict]:
    """
    Пакетное предсказание.
    
    Args:
        model_path: Путь к модели
        input_data: Входные данные
        
    Returns:
        Список с предсказаниями
    """
    predictor = ModelPredictor(model_path)
    return predictor.predict_batch(input_data)


if __name__ == "__main__":
    example_input = {
        'Баллы ЕГЭ': 70.0,
        'exam_month': 12,
        'exam_year': 2023,
        'exam_day_of_week': 2,
        'Вид контроля_encoded': 0,
        'Состояние_encoded': 1,
        'degree_encoded': 0,
        'Подразделение_encoded': 2
    }
    
    prediction = predict_single("models/model_v1.pkl", example_input)
    print(f"Predicted mark: {prediction}")
