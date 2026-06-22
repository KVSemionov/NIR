import json, random

subjects = [
    "EGE_Английский_язык",
    "EGE_Информатика_и_ИКТ",
    "EGE_История",
    "EGE_Математика",
    "EGE_Обществознание",
    "EGE_Русский_язык",
    "EGE_Физика",
    "EGE_Химия"
]

students = []

for _ in range(100):
    student = {
        "Подразделение": "Институт информационных технологий и компьютерных наук",
        "Курс_по_порядку": random.randint(1, 4),
        "history_mean": round(random.uniform(3.4, 4.8), 1),
        "last_mark": round(random.uniform(3.5, 5.0), 1)
    }
    
    chosen = random.sample(subjects, random.randint(3,4))
    
    for subj in subjects:
        if subj in chosen:
            student[subj] = random.randint(60, 95)
        else:
            student[subj] = 0
    
    students.append(student)

file_path = "/mnt/data/students_100.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(students, f, ensure_ascii=False, indent=2)

file_path