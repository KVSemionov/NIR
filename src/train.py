import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import optuna
from .logger import setup_logger, log_model_metrics
from .preprocess import DataPreprocessor
from .features import FeatureEngineer


class ModelTrainer:
    """Класс для обучения и валидации модели."""
    
    def __init__(self, model_type: str = "random_forest", logger=None):
        self.model_type = model_type
        self.logger = logger or setup_logger("train")
        self.model = None
        self.feature_names = None
        self.metrics = {}
    
    def create_model(self, params: Optional[Dict] = None):
        """
        Создание модели с заданными параметрами.
        
        Args:
            params: Параметры модели
            
        Returns:
            Модель scikit-learn
        """
        if self.model_type == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError as exc:
                raise ImportError(
                    "LightGBM is required for model_type='lightgbm'. "
                    "Install dependencies with: python -m pip install -r requirements.txt"
                ) from exc

            default_params = {
                'n_estimators': 1500,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': -1,
                'random_state': 42,
                'verbosity': -1
            }
            params = {**default_params, **(params or {})}
            self.model = lgb.LGBMRegressor(**params)

        elif self.model_type == "random_forest":
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
            params = {**default_params, **(params or {})}
            self.model = RandomForestRegressor(**params)
            
        elif self.model_type == "gradient_boosting":
            default_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42
            }
            params = {**default_params, **(params or {})}
            self.model = GradientBoostingRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.logger.info(f"Created {self.model_type} model with params: {params}")
        return self.model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Обучение модели.
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающая целевая переменная
        """
        self.logger.info(f"Training {self.model_type} model on {len(X_train)} samples")
        
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        
        self.logger.info("Model training completed")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Оценка модели на тестовых данных.
        
        Args:
            X_test: Тестовые признаки
            y_test: Тестовая целевая переменная
            
        Returns:
            Словарь с метриками
        """
        self.logger.info("Evaluating model on test set")
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        self.metrics = metrics
        
        self.logger.info(f"Test metrics: {metrics}")
        log_model_metrics(self.logger, metrics, model_version="v1")
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Кросс-валидация модели.
        
        Args:
            X: Признаки
            y: Целевая переменная
            cv: Количество фолдов
            
        Returns:
            Словарь с метриками кросс-валидации
        """
        self.logger.info(f"Running {cv}-fold cross validation")
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_squared_error')
        
        cv_metrics = {
            'cv_rmse_mean': np.sqrt(-scores.mean()),
            'cv_rmse_std': np.sqrt(-scores).std()
        }
        
        self.logger.info(f"Cross-validation metrics: {cv_metrics}")
        
        return cv_metrics
    
    def hyperparameter_tuning(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        n_trials: int = 50
    ) -> Dict:
        """
        Подбор гиперпараметров с помощью Optuna.
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающая целевая переменная
            n_trials: Количество итераций
            
        Returns:
            Лучшие параметры
        """
        self.logger.info(f"Starting hyperparameter tuning with {n_trials} trials")
        
        def objective(trial):
            if self.model_type == "random_forest":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                    'random_state': 42,
                    'n_jobs': -1
                }
            elif self.model_type == "gradient_boosting":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'random_state': 42
                }
            
            model = self.create_model(params)
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
            return np.sqrt(-scores.mean())
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best CV RMSE: {study.best_value:.4f}")
        
        return best_params
    
    def save_model(self, filepath: str, model_version: str = "v1"):
        """
        Сохранение модели и метаданных.
        
        Args:
            filepath: Путь для сохранения модели
            model_version: Версия модели
        """
        model_dir = Path(filepath).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'metrics': self.metrics,
            'version': model_version
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
        
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'version': model_version
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, filepath: str):
        """
        Загрузка модели.
        
        Args:
            filepath: Путь к файлу модели
        """
        self.logger.info(f"Loading model from {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.metrics = model_data.get('metrics', {})
        
        self.logger.info(f"Model loaded. Type: {self.model_type}, Version: {model_data.get('version', 'unknown')}")


def train_pipeline(
    data_path: str,
    model_save_path: str,
    model_type: str = "random_forest",
    tune_hyperparams: bool = False,
    model_version: str = "v1"
):
    """
    Полный пайплайн обучения модели.
    
    Args:
        data_path: Путь к данным
        model_save_path: Путь для сохранения модели
        model_type: Тип модели
        tune_hyperparams: Подбирать ли гиперпараметры
        model_version: Версия модели
    """
    logger = setup_logger("train_pipeline")
    
    logger.info("Starting training pipeline")
    
    preprocessor = DataPreprocessor(logger)
    feature_engineer = FeatureEngineer(logger)
    
    df = preprocessor.preprocess(data_path)
    df = feature_engineer.engineer_features(df)
    
    feature_list = feature_engineer.get_feature_list(df)
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_data(df, feature_cols=feature_list)
    
    trainer = ModelTrainer(model_type=model_type, logger=logger)
    
    if tune_hyperparams:
        best_params = trainer.hyperparameter_tuning(X_train, y_train, n_trials=50)
        trainer.create_model(best_params)
    else:
        trainer.create_model()
    
    trainer.train(X_train, y_train)
    
    metrics = trainer.evaluate(X_test, y_test)
    cv_metrics = trainer.cross_validate(X_train, y_train)
    
    trainer.save_model(model_save_path, model_version)
    
    logger.info("Training pipeline completed successfully")
    
    return metrics, cv_metrics


def train_notebook_pipeline(
    data_path: str,
    model_save_path: str,
    model_type: str = "lightgbm",
    model_version: str = "v1_notebook_mlops",
):
    """
    MLOps-версия исходного notebook model_2025_final_v3.ipynb.
    Сохраняет исходную идею: GPA по семестрам + EGE_* + history_mean + last_mark,
    но оформляет ее как воспроизводимый training pipeline с сохранением модели.
    """
    logger = setup_logger("notebook_train_pipeline")
    logger.info("Starting notebook-based MLOps training pipeline")

    preprocessor = DataPreprocessor(logger)
    feature_engineer = FeatureEngineer(logger)

    raw_df = preprocessor.load_data(data_path)
    final_train_set = feature_engineer.build_notebook_training_dataset(raw_df)

    X = final_train_set.drop(columns=["ld_number", "target"])
    y = final_train_set["target"]

    X = feature_engineer.clean_column_names(X)
    categorical_features = [col for col in ["Подразделение"] if col in X.columns]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trainer = ModelTrainer(model_type=model_type, logger=logger)
    base_model = trainer.create_model()

    preprocessor_pipeline = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
    )

    trainer.model = Pipeline([
        ("preprocess", preprocessor_pipeline),
        ("model", base_model),
    ])
    trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)
    cv_metrics = trainer.cross_validate(X_train, y_train)
    trainer.save_model(model_save_path, model_version)

    logger.info("Notebook-based MLOps training pipeline completed successfully")
    return metrics, cv_metrics


if __name__ == "__main__":
    train_notebook_pipeline(
        data_path="data/marks_25_hash.csv",
        model_save_path="models/model_v1.pkl",
        model_type="lightgbm",
        model_version="v1_notebook_mlops"
    )
