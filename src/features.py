import pandas as pd
import numpy as np
import re
from typing import Dict, List
from .logger import setup_logger


class FeatureEngineer:
    """Класс для генерации признаков на основе исторических данных студентов."""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("features")

    def normalize_semester_name(self, sem_string):
        """
        Нормализация названия семестра из исходного notebook model_2025_final_v3.ipynb.
        Пример: "2023 - 2024II полугодие" -> "23_24_semestr2".
        """
        if pd.isna(sem_string):
            return sem_string

        value = str(sem_string).strip()

        if "полугодие" in value:
            years = re.findall(r"20(\d{2})", value)
            year_part = f"{years[0]}_{years[1]}" if len(years) >= 2 else "unknown"
            sem_part = "semestr1" if "I полугодие" in value and "II" not in value else "semestr2"
            return f"{year_part}_{sem_part}"

        if re.match(r"\d{2}_\d{2}_semestr\d", value):
            return value

        return value

    def build_notebook_training_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Перенос ключевой логики из notebook:
        - ЕГЭ разворачивается в отдельные признаки EGE_*
        - оценки переводятся в GPA по студенту и семестру
        - из истории семестров создаются признаки history_mean и last_mark
        - один студент дает несколько обучающих примеров: прошлые семестры -> следующий GPA
        """
        self.logger.info("Building notebook-compatible GPA training dataset")

        data = df.copy()
        data["marks_final"] = data["marks_final"].str.lower().fillna("неудовлетворительно")
        neg_marks = ["не зачтено", "не допущен", "неявка", "неявка по ув.причине"]
        data["marks_final"] = data["marks_final"].replace(neg_marks, "неудовлетворительно")

        marks_map = {
            "отлично": 5,
            "хорошо": 4,
            "удовлетворительно": 3,
            "зачтено": 4,
            "неудовлетворительно": 2,
        }
        data["marks_final_numeric"] = data["marks_final"].map(marks_map)
        data = data.dropna(subset=["ld_number", "marks_final_numeric"])

        ege_pivot = data.pivot_table(
            index="ld_number",
            columns="Предметы ЕГЭ",
            values="Баллы ЕГЭ",
            aggfunc="max",
        ).fillna(0)
        ege_pivot.columns = [f"EGE_{col}" for col in ege_pivot.columns]

        data["date_normalized"] = data["date"].apply(self.normalize_semester_name)
        grouped_marks = (
            data.groupby(["ld_number", "date_normalized"])["marks_final_numeric"]
            .mean()
            .reset_index()
        )

        sem_gpa = grouped_marks.pivot(
            index="ld_number",
            columns="date_normalized",
            values="marks_final_numeric",
        ).fillna(0)
        sem_gpa.columns = [f"gpa_{col}" for col in sem_gpa.columns]

        static_features = data.groupby("ld_number").agg({
            "Подразделение": "first",
            "Структурное подразделение": "first",
            "Курс по порядку": "max",
        }).reset_index()

        dataset = static_features.merge(sem_gpa, on="ld_number", how="left")
        dataset = dataset.merge(ege_pivot, on="ld_number", how="left").fillna(0)

        gpa_columns = sorted(
            [col for col in dataset.columns if col.startswith("gpa_")],
            key=self._semester_sort_key,
        )

        train_rows = []
        for i in range(2, len(gpa_columns)):
            target_sem = gpa_columns[i]
            past_sems = gpa_columns[:i]
            temp_df = dataset[dataset[target_sem] > 0].copy()
            if temp_df.empty:
                continue

            temp_df["target"] = temp_df[target_sem]
            temp_df["history_mean"] = temp_df[past_sems].replace(0, np.nan).mean(axis=1)
            temp_df["last_mark"] = temp_df[past_sems[-1]]

            cols_to_keep = (
                ["ld_number", "Подразделение", "Курс по порядку", "history_mean", "last_mark", "target"]
                + [col for col in dataset.columns if col.startswith("EGE_")]
            )
            train_rows.append(temp_df[cols_to_keep])

        if not train_rows:
            raise ValueError("Not enough semester GPA columns to build notebook-compatible dataset")

        final_train_set = pd.concat(train_rows, ignore_index=True)
        final_train_set = final_train_set.dropna(subset=["history_mean", "last_mark"])

        self.logger.info(f"Notebook-compatible dataset created: {final_train_set.shape}")
        return final_train_set

    def _semester_sort_key(self, column_name: str):
        match = re.search(r"gpa_(\d{2})_(\d{2})_semestr(\d)", column_name)
        if not match:
            return (9999, 9)
        return (int(match.group(1)), int(match.group(3)))

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистка названий признаков для моделей бустинга, как в notebook."""
        cleaned = df.copy()
        cleaned.columns = [re.sub(r"[^\w\s]", "_", str(col)) for col in cleaned.columns]
        cleaned.columns = [col.replace(" ", "_").replace("__", "_") for col in cleaned.columns]
        return cleaned
    
    def calculate_student_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Вычисление агрегированных показателей для каждого студента.
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с агрегированными признаками
        """
        self.logger.info("Calculating student aggregate features")
        
        student_features = df.groupby('ld_number').agg({
            'marks_numeric': ['mean', 'std', 'min', 'max', 'count'],
            'Баллы ЕГЭ': 'first'
        }).reset_index()
        
        student_features.columns = [
            'ld_number',
            'avg_mark',
            'std_mark',
            'min_mark',
            'max_mark',
            'total_subjects',
            'ege_score'
        ]
        
        student_features['std_mark'] = student_features['std_mark'].fillna(0)
        
        df = df.merge(student_features, on='ld_number', how='left')
        
        self.logger.info(f"Added {len(student_features.columns) - 1} student aggregate features")
        
        return df
    
    def calculate_subject_difficulty(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Вычисление сложности предметов на основе средней оценки.
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с признаком сложности предмета
        """
        self.logger.info("Calculating subject difficulty features")
        
        subject_stats = df.groupby('subject')['marks_numeric'].agg(['mean', 'count']).reset_index()
        subject_stats.columns = ['subject', 'subject_avg_mark', 'subject_count']
        
        df = df.merge(subject_stats, on='subject', how='left')
        
        df['subject_difficulty'] = 5 - df['subject_avg_mark']
        
        self.logger.info("Added subject difficulty feature")
        
        return df
    
    def calculate_semester_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Вычисление трендов успеваемости по семестрам.
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с признаками трендов
        """
        self.logger.info("Calculating semester trend features")
        
        df = df.sort_values(['ld_number', 'Дата занятия'])
        
        df['prev_mark'] = df.groupby('ld_number')['marks_numeric'].shift(1)
        df['mark_diff'] = df['marks_numeric'] - df['prev_mark']
        
        df['mark_diff'] = df['mark_diff'].fillna(0)
        
        self.logger.info("Added semester trend features")
        
        return df
    
    def calculate_control_type_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Статистика по типам контроля (зачет, экзамен и т.д.).
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame со статистикой по типам контроля
        """
        self.logger.info("Calculating control type statistics")
        
        control_stats = df.groupby(['ld_number', 'Вид контроля']).size().unstack(fill_value=0)
        
        control_stats.columns = [f'control_{col}' for col in control_stats.columns]
        
        df = df.merge(control_stats, on='ld_number', how='left')
        
        self.logger.info(f"Added {len(control_stats.columns)} control type features")
        
        return df
    
    def calculate_success_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Вычисление процента успешной сдачи предметов.
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с признаком success_rate
        """
        self.logger.info("Calculating success rate features")
        
        df['is_success'] = (df['marks_numeric'] >= 3).astype(int)
        
        student_success = df.groupby('ld_number')['is_success'].agg(['mean', 'sum']).reset_index()
        student_success.columns = ['ld_number', 'success_rate', 'total_passed']
        
        df = df.merge(student_success, on='ld_number', how='left')
        
        self.logger.info("Added success rate features")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Полный пайплайн генерации признаков.
        
        Args:
            df: Предобработанный DataFrame
            
        Returns:
            DataFrame с новыми признаками
        """
        self.logger.info("Starting feature engineering pipeline")
        
        df = self.calculate_student_aggregates(df)
        df = self.calculate_subject_difficulty(df)
        df = self.calculate_semester_trends(df)
        df = self.calculate_control_type_stats(df)
        df = self.calculate_success_rate(df)
        
        self.logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")
        
        return df
    
    def get_feature_list(self, df: pd.DataFrame) -> List[str]:
        """
        Получение списка признаков для обучения модели.
        
        Args:
            df: DataFrame с признаками
            
        Returns:
            Список названий признаков
        """
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        exclude_cols = [
            'ld_number',
            'marks_numeric',
            'marks_final_numeric',
            'is_success',
            'subject_avg_mark',
            'subject_count'
        ]
        
        feature_list = [col for col in numeric_features if col not in exclude_cols]
        
        self.logger.info(f"Selected {len(feature_list)} features for training")
        
        return feature_list
