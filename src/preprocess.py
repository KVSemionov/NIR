import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from .logger import setup_logger


class DataPreprocessor:
    """Класс для предобработки данных об успеваемости студентов."""
    
    MARKS_MAPPING = {
        'Отлично': 5,
        'Хорошо': 4,
        'Удовлетворительно': 3,
        'Неудовлетворительно': 2,
        'зачтено': 1,
        'не зачтено': 0
    }
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("preprocess")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Загрузка данных из CSV файла.
        
        Args:
            filepath: Путь к файлу с данными
            
        Returns:
            DataFrame с данными
        """
        self.logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath, sep=',')
        self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df 
        
    def normalize_sem_name(sem_string):
     if pd.isna(sem_string):
        return sem_string

     s = str(sem_string).strip()

     # 1. Обработка формата "2023 - 2024II полугодие"
     if 'полугодие' in s:
        # Извлекаем годы (ищем две цифры после 20)
        years = re.findall(r'20(\d{2})', s)
        if len(years) >= 2:
            year_part = f"{years[0]}_{years[1]}"
        else:
            year_part = "unknown"

        # Определяем семестр
        sem_part = "semestr1" if "I полугодие" in s and "II" not in s else "semestr2"
        return f"{year_part}_{sem_part}"

     # 2. Обработка формата "22_23_semestr2" (оставляем как есть, если подходит)
     if re.match(r'\d{2}_\d{2}_semestr\d', s):
        return s

     return s # Если формат совсем другой

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Очистка данных: удаление дубликатов, обработка пропусков.
        
        Args:
            df: Исходный DataFrame
            
        Returns:
            Очищенный DataFrame
        """
        self.logger.info("Starting data cleaning")
        
        initial_rows = len(df)
        
        df = df.drop_duplicates()
        self.logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        df = df.dropna(subset=['ld_number', 'subject'])
        self.logger.info(f"Removed rows with missing key fields")
        
        return df
    
    def convert_marks_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразование текстовых оценок в числовые значения.
        
        Args:
            df: DataFrame с текстовыми оценками
            
        Returns:
            DataFrame с числовыми оценками
        """
        self.logger.info("Converting marks to numeric values")
        
        df['marks_numeric'] = df['marks'].map(self.MARKS_MAPPING)
        df['marks_final_numeric'] = df['marks_final'].map(self.MARKS_MAPPING)
        
        missing_marks = df['marks_numeric'].isna().sum()
        self.logger.info(f"Converted marks, {missing_marks} rows have unknown marks")
        
        return df
    
    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Парсинг дат из строковых форматов.
        
        Args:
            df: DataFrame с датами в строковом формате
            
        Returns:
            DataFrame с преобразованными датами
        """
        self.logger.info("Parsing dates")
        
        date_columns = ['Дата занятия', 'Дата начала обучения', 'Дата окончания обучения (по приказу)']
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%d.%m.%Y', errors='coerce')
        
        return df
    
    def extract_features_from_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение признаков из дат (месяц, год, день недели).
        
        Args:
            df: DataFrame с датами
            
        Returns:
            DataFrame с новыми признаками
        """
        self.logger.info("Extracting features from dates")
        
        if 'Дата занятия' in df.columns:
            df['exam_month'] = df['Дата занятия'].dt.month
            df['exam_year'] = df['Дата занятия'].dt.year
            df['exam_day_of_week'] = df['Дата занятия'].dt.dayofweek
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Кодирование категориальных признаков.
        
        Args:
            df: DataFrame с категориальными признаками
            
        Returns:
            DataFrame с закодированными признаками
        """
        self.logger.info("Encoding categorical features")
        
        categorical_cols = ['Вид контроля', 'Состояние', 'degree', 'Подразделение']
        
        for col in categorical_cols:
            if col in df.columns:
                df[col + '_encoded'] = pd.factorize(df[col])[0]
        
        return df
    
    def preprocess(self, filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Полный пайплайн предобработки данных.
        
        Args:
            filepath: Путь к исходному файлу
            
        Returns:
            Кортеж (X, y) - признаки и целевая переменная
        """
        df = self.load_data(filepath)
        df = self.clean_data(df)
        df = self.convert_marks_to_numeric(df)
        df = self.parse_dates(df)
        df = self.extract_features_from_dates(df)
        df = self.encode_categorical(df)
        
        self.logger.info("Preprocessing completed")
        
        return df
    
    def prepare_train_test_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'marks_final_numeric',
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Разделение данных на обучающую и тестовую выборки.
        
        Args:
            df: Предобработанный DataFrame
            target_col: Целевая колонка
            feature_cols: Список признаков для включения в выборку
            test_size: Размер тестовой выборки
            random_state: Random state для воспроизводимости
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        from sklearn.model_selection import train_test_split
        
        if feature_cols is None:
            feature_cols = [
                'Баллы ЕГЭ',
                'exam_month',
                'exam_year',
                'exam_day_of_week',
                'Вид контроля_encoded',
                'Состояние_encoded',
                'degree_encoded',
                'Подразделение_encoded'
            ]
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].fillna(0)
        y = df[target_col].fillna(0)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
