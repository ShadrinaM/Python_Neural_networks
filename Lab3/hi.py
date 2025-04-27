import random

# Исходные данные
raw_data = [
    ["Варданян", 1, 1, 60, 79, 60, 72, 63, 1.00],
    ["Горбунов", 1, 0, 60, 61, 30, 5, 17, 0.00],
    ["Гуменюк", 0, 0, 60, 61, 30, 66, 58, 0.00],
    ["Егоров", 1, 1, 85, 78, 72, 70, 85, 1.25],
    ["Захарова", 0, 1, 65, 78, 60, 67, 65, 1.00],
    ["Иванова", 0, 1, 60, 78, 77, 81, 60, 1.25],
    ["Ишонина", 0, 1, 55, 79, 56, 69, 72, 0.00],
    ["Климчук", 1, 0, 55, 56, 50, 56, 60, 0.00],
    ["Лисовский", 1, 0, 55, 60, 21, 64, 50, 0.00],
    ["Нетреба", 1, 0, 60, 56, 30, 16, 17, 0.00],
    ["Остапова", 0, 1, 85, 89, 85, 92, 85, 1.75],
    ["Пашкова", 0, 1, 60, 88, 76, 66, 60, 1.25],
    ["Попов", 1, 0, 55, 64, 0, 9, 50, 0.00],
    ["Сазон", 0, 1, 80, 83, 62, 72, 72, 1.25],
    ["Степоненко", 1, 0, 55, 10, 3, 8, 50, 0.00],
    ["Терентьева", 0, 1, 60, 67, 57, 64, 50, 0.00],
    ["Титов", 1, 1, 75, 98, 86, 82, 85, 1.50],
    ["Чернова", 0, 1, 85, 85, 81, 85, 72, 1.25],
    ["Четкин", 1, 1, 80, 56, 50, 69, 50, 0.00],
    ["Шевченко", 1, 0, 55, 60, 30, 8, 60, 0.00]
]

# Нормализация данных (кроме фамилии и стипендии)
def normalize(data):
    numeric_data = [row[1:-1] for row in data]
    normalized = []
    for col in range(len(numeric_data[0])):
        col_data = [row[col] for row in numeric_data]
        min_val = min(col_data)
        max_val = max(col_data)
        norm_col = [(x - min_val) / (max_val - min_val) if max_val != min_val else 0.0 for x in col_data]
        normalized.append(norm_col)
    normalized = list(map(list, zip(*normalized)))
    final = [[data[i][0]] + normalized[i] + [data[i][-1]] for i in range(len(data))]
    return final

# Функция Евклидова расстояния
def euclidean(a, b):
    return sum((x - y)**2 for x, y in zip(a, b))**0.5

# Параметры сети
input_size = 7
output_size = 4

# Нормализация
normalized_students = normalize(raw_data)

# Инициализация весов из случайных студентов
initial_indices = random.sample(range(len(normalized_students)), output_size)
weights = [normalized_students[idx][1:-1] for idx in initial_indices]

# Обучение сети
learning_rate = 0.3
for epoch in range(6):
    for _ in range(20):
        for sample in normalized_students:
            inputs = sample[1:-1]
            distances = [euclidean(inputs, w) for w in weights]
            winner_idx = distances.index(min(distances))
            for j in range(input_size):
                weights[winner_idx][j] += learning_rate * (inputs[j] - weights[winner_idx][j])
    learning_rate -= 0.05

# Кластеризация
clusters = [[] for _ in range(output_size)]
for idx, sample in enumerate(normalized_students):
    inputs = sample[1:-1]
    distances = [euclidean(inputs, w) for w in weights]
    winner_idx = distances.index(min(distances))
    clusters[winner_idx].append((idx + 1, raw_data[idx]))

# Красивый вывод
header = f"{'№':<3} {'Фамилия':<10} {'Пол':<5} {'Зачеты':<7} {'Ист.':<5} {'Граф.':<6} {'Матем.':<7} {'Хим.':<5} {'Физ.':<5} {'Стип.':<5}"

for i, cluster in enumerate(clusters):
    print(f"\nКластер {i+1}:")
    print(header)
    print("-"*70)
    for student_idx, student_data in cluster:
        gender = "М" if student_data[1] == 1 else "Ж"
        zachety = "Да" if student_data[2] == 1 else "Нет"
        print(f"{student_idx:<3} {student_data[0]:<10} {gender:<5} {zachety:<7} {student_data[3]:<5} {student_data[4]:<6} {student_data[5]:<7} {student_data[6]:<5} {student_data[7]:<5} {student_data[8]:<5}")