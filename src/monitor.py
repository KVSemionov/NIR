import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from scipy import stats
from .logger import setup_logger


class ModelMonitor:
    """Класс для мониторинга качества модели и дрейфа данных."""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("monitor")
        self.baseline_stats = {}
        self.prediction_history = []
    
    def calculate_baseline_statistics(self, df: pd.DataFrame, feature_cols: List[str]):
        """
        Вычисление базовых статистик для обучающих данных.
        
        Args:
            df: Обучающие данные
            feature_cols: Список признаков для мониторинга
        """
        self.logger.info("Calculating baseline statistics")
        
        for col in feature_cols:
            if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                self.baseline_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median(),
                    'q25': df[col].quantile(0.25),
                    'q75': df[col].quantile(0.75)
                }
        
        self.logger.info(f"Baseline statistics calculated for {len(self.baseline_stats)} features")
    
    def detect_data_drift(
        self, 
        current_data: pd.DataFrame, 
        threshold: float = 0.05
    ) -> Dict[str, Dict]:
        """
        Обнаружение дрейфа данных с помощью статистических тестов.
        
        Args:
            current_data: Текущие данные
            threshold: Порог значимости для тестов
            
        Returns:
            Словарь с результатами тестов на дрейф
        """
        self.logger.info("Detecting data drift")
        
        drift_results = {}
        
        for feature, baseline in self.baseline_stats.items():
            if feature not in current_data.columns:
                continue
            
            current_values = current_data[feature].dropna()
            
            if len(current_values) < 30:
                self.logger.warning(f"Not enough samples for drift detection on {feature}")
                continue
            
            z_score = abs((current_values.mean() - baseline['mean']) / baseline['std']) if baseline['std'] > 0 else 0
            
            ks_stat, ks_pvalue = stats.ks_2samp(
                current_values, 
                np.random.normal(baseline['mean'], baseline['std'], len(current_values))
            )
            
            drift_detected = ks_pvalue < threshold or z_score > 2
            
            drift_results[feature] = {
                'drift_detected': drift_detected,
                'z_score': z_score,
                'ks_pvalue': ks_pvalue,
                'current_mean': current_values.mean(),
                'baseline_mean': baseline['mean'],
                'mean_diff': current_values.mean() - baseline['mean']
            }
            
            if drift_detected:
                self.logger.warning(f"Data drift detected for {feature}: z_score={z_score:.2f}, ks_pvalue={ks_pvalue:.4f}")
        
        return drift_results
    
    def track_prediction(
        self, 
        input_data: Dict, 
        prediction: float, 
        actual: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Отслеживание предсказания с возможностью сравнения с фактическим значением.
        
        Args:
            input_data: Входные данные
            prediction: Предсказание модели
            actual: Фактическое значение (если доступно)
            timestamp: Время предсказания
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        record = {
            'timestamp': timestamp,
            'input_data': input_data,
            'prediction': prediction,
            'actual': actual,
            'error': abs(prediction - actual) if actual is not None else None
        }
        
        self.prediction_history.append(record)
        
        if actual is not None:
            self.logger.info(f"Prediction tracked | Predicted: {prediction:.2f}, Actual: {actual:.2f}, Error: {record['error']:.2f}")
    
    def calculate_model_performance_metrics(self) -> Dict[str, float]:
        """
        Вычисление метрик качества модели на основе истории предсказаний.
        
        Returns:
            Словарь с метриками
        """
        predictions_with_actual = [p for p in self.prediction_history if p['actual'] is not None]
        
        if len(predictions_with_actual) == 0:
            self.logger.warning("No predictions with actual values available")
            return {}
        
        predictions = np.array([p['prediction'] for p in predictions_with_actual])
        actuals = np.array([p['actual'] for p in predictions_with_actual])
        
        metrics = {
            'mae': np.mean(np.abs(predictions - actuals)),
            'rmse': np.sqrt(np.mean((predictions - actuals) ** 2)),
            'mean_error': np.mean(predictions - actuals),
            'total_predictions': len(self.prediction_history),
            'predictions_with_actual': len(predictions_with_actual)
        }
        
        self.logger.info(f"Model performance metrics: {metrics}")
        
        return metrics
    
    def detect_performance_degradation(
        self, 
        baseline_metrics: Dict[str, float],
        degradation_threshold: float = 0.2
    ) -> Dict[str, bool]:
        """
        Обнаружение деградации качества модели.
        
        Args:
            baseline_metrics: Базовые метрики
            degradation_threshold: Порог деградации (20% по умолчанию)
            
        Returns:
            Словарь с флагами деградации по метрикам
        """
        current_metrics = self.calculate_model_performance_metrics()
        
        if not current_metrics:
            return {}
        
        degradation_results = {}
        
        for metric in ['mae', 'rmse']:
            if metric in baseline_metrics and metric in current_metrics:
                relative_increase = (current_metrics[metric] - baseline_metrics[metric]) / baseline_metrics[metric]
                degradation_detected = relative_increase > degradation_threshold
                
                degradation_results[metric] = {
                    'degradation_detected': degradation_detected,
                    'baseline': baseline_metrics[metric],
                    'current': current_metrics[metric],
                    'relative_increase': relative_increase
                }
                
                if degradation_detected:
                    self.logger.warning(
                        f"Performance degradation detected for {metric}: "
                        f"baseline={baseline_metrics[metric]:.4f}, "
                        f"current={current_metrics[metric]:.4f}, "
                        f"increase={relative_increase:.2%}"
                    )
        
        return degradation_results
    
    def save_monitoring_report(self, filepath: str):
        """
        Сохранение отчета мониторинга в файл.
        
        Args:
            filepath: Путь для сохранения отчета
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'baseline_stats': self.baseline_stats,
            'performance_metrics': self.calculate_model_performance_metrics(),
            'total_predictions': len(self.prediction_history)
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Monitoring report saved to {filepath}")
    
    def load_monitoring_report(self, filepath: str):
        """
        Загрузка отчета мониторинга из файла.
        
        Args:
            filepath: Путь к файлу отчета
        """
        import json
        with open(filepath, 'r') as f:
            report = json.load(f)
        
        self.baseline_stats = report.get('baseline_stats', {})
        self.logger.info(f"Monitoring report loaded from {filepath}")


class PredictionLogger:
    """Класс для логирования предсказаний в отдельный файл."""
    
    def __init__(self, log_file: str = "logs/predictions.log", logger=None):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger or setup_logger("prediction_logger")
    
    def log(
        self,
        input_data: Dict,
        prediction: float,
        model_version: str = "v1",
        actual: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Логирование предсказания в файл.
        
        Args:
            input_data: Входные данные
            prediction: Предсказание
            model_version: Версия модели
            actual: Фактическое значение
            metadata: Дополнительная информация
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version,
            'input_data': input_data,
            'prediction': prediction,
            'actual': actual,
            'metadata': metadata or {}
        }
        
        import json
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        self.logger.info(f"Prediction logged to {self.log_file}")
    
    def read_logs(self, n_last: int = 100) -> List[Dict]:
        """
        Чтение последних N записей из лога.
        
        Args:
            n_last: Количество последних записей
            
        Returns:
            Список записей
        """
        if not self.log_file.exists():
            return []
        
        import json
        logs = []
        
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        return logs[-n_last:] if logs else []
