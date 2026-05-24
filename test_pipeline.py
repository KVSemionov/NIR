"""
Скрипт для тестирования всего ML пайплайна.
Запустите: python test_pipeline.py
"""

import sys
from pathlib import Path

print("=" * 60)
print("Testing ML Student Performance Prediction Pipeline")
print("=" * 60)

# Шаг 1: Проверка структуры проекта
print("\n1. Checking project structure...")
required_dirs = ['data', 'models', 'src', 'api', 'logs']
for dir_name in required_dirs:
    if Path(dir_name).exists():
        print(f"   ✓ {dir_name}/ exists")
    else:
        print(f"   ✗ {dir_name}/ missing")

# Шаг 2: Проверка наличия данных
print("\n2. Checking data files...")
data_file = Path('data/marks_25_hash.csv')
if data_file.exists():
    print(f"   ✓ Data file exists: {data_file}")
else:
    print(f"   ✗ Data file missing: {data_file}")

# Шаг 3: Проверка модулей
print("\n3. Checking Python modules...")
modules = ['logger', 'preprocess', 'features', 'train', 'predict', 'monitor']
for module in modules:
    module_path = Path(f'src/{module}.py')
    if module_path.exists():
        print(f"   ✓ src/{module}.py exists")
    else:
        print(f"   ✗ src/{module}.py missing")

# Шаг 4: Проверка API
print("\n4. Checking API module...")
api_file = Path('api/app.py')
if api_file.exists():
    print(f"   ✓ api/app.py exists")
else:
    print(f"   ✗ api/app.py missing")

# Шаг 5: Проверка зависимостей
print("\n5. Checking dependencies...")
try:
    import pandas
    print("   ✓ pandas installed")
except ImportError:
    print("   ✗ pandas not installed")

try:
    import numpy
    print("   ✓ numpy installed")
except ImportError:
    print("   ✗ numpy not installed")

try:
    import sklearn
    print("   ✓ scikit-learn installed")
except ImportError:
    print("   ✗ scikit-learn not installed")

try:
    import fastapi
    print("   ✓ fastapi installed")
except ImportError:
    print("   ✗ fastapi not installed")

# Шаг 6: Тест обучения модели
print("\n6. Testing model training...")
try:
    from src.train import train_notebook_pipeline
    print("   ✓ Import successful")
    
    print("   Starting training pipeline...")
    metrics, cv_metrics = train_notebook_pipeline(
        data_path="data/marks_25_hash.csv",
        model_save_path="models/model_v1.pkl",
        model_type="lightgbm",
        model_version="v1_notebook_mlops"
    )
    
    print(f"   ✓ Model trained successfully")
    print(f"   Test RMSE: {metrics.get('rmse', 'N/A'):.4f}")
    print(f"   Test R²: {metrics.get('r2', 'N/A'):.4f}")
    print(f"   CV RMSE: {cv_metrics.get('cv_rmse_mean', 'N/A'):.4f}")
    
except Exception as e:
    print(f"   ✗ Training failed: {e}")

# Шаг 7: Тест предсказания
print("\n7. Testing prediction...")
try:
    from src.predict import predict_single
    
    example_input = {
        'Подразделение': 'Институт информационных технологий и компьютерных наук',
        'Курс_по_порядку': 4,
        'history_mean': 4.2,
        'last_mark': 4.5,
        'EGE_Информатика_и_ИКТ': 80,
        'EGE_Математика': 76,
        'EGE_Русский_язык': 85
    }
    
    prediction = predict_single("models/model_v1.pkl", example_input)
    print(f"   ✓ Prediction successful: {prediction:.2f}")
    
except Exception as e:
    print(f"   ✗ Prediction failed: {e}")

# Шаг 8: Проверка логов
print("\n8. Checking logs...")
log_files = list(Path('logs').glob('*.log'))
if log_files:
    print(f"   ✓ Found {len(log_files)} log file(s)")
    for log_file in log_files:
        print(f"     - {log_file.name}")
else:
    print("   ℹ No log files yet (will be created on first run)")

print("\n" + "=" * 60)
print("Pipeline test completed!")
print("=" * 60)
print("\nNext steps:")
print("1. Run: python -m api.app  (to start the API server)")
print("2. Visit: http://localhost:8000/docs  (API documentation)")
print("3. Test endpoints via Swagger UI")
