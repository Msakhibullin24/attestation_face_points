import pandas as pd

# Загружаем обучающие данные
df = pd.read_csv('training.csv')

# Рассчитываем процент пропущенных значений для каждого столбца
missing_values = df.isnull().sum()
total_rows = len(df)
missing_percentage = (missing_values / total_rows) * 100

# Выводим результаты
print("Процент пропущенных значений для каждой ключевой точки:")
for col, percentage in missing_percentage.items():
    if percentage > 0:
        print(f"{col}: {percentage:.2f}%")
