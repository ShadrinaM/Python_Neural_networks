# import random

# # === ШАГ 1. Подготовка данных ===

# # Список студентов (в виде словаря: имя, пол, зачёты, история, графика, математика, химия, физика, стипендия)
# raw_data = [
#     ["Варданян", 1, 1, 60, 79, 60, 72, 63, 1.00],
#     ["Горбунов", 1, 0, 60, 61, 30, 5, 17, 0.00],
#     ["Гуменюк", 0, 0, 60, 61, 30, 66, 58, 0.00],
#     ["Егоров", 1, 1, 85, 78, 72, 70, 85, 1.25],
#     ["Захарова", 0, 1, 65, 78, 60, 67, 65, 1.00],
#     ["Иванова", 0, 1, 60, 78, 77, 81, 60, 1.25],
#     ["Ишонина", 0, 1, 55, 79, 56, 69, 72, 0.00],
#     ["Климчук", 1, 0, 55, 56, 50, 56, 60, 0.00],
#     ["Лисовский", 1, 0, 55, 60, 21, 64, 50, 0.00],
#     ["Нетреба", 1, 0, 60, 56, 30, 16, 17, 0.00],
#     ["Остапова", 0, 1, 85, 89, 85, 92, 85, 1.75],
#     ["Пашкова", 0, 1, 60, 88, 76, 66, 60, 1.25],
#     ["Попов", 1, 0, 55, 64, 0, 9, 50, 0.00],
#     ["Сазон", 0, 1, 80, 83, 62, 72, 72, 1.25],
#     ["Степоненко", 1, 0, 55, 10, 3, 8, 50, 0.00],
#     ["Терентьева", 0, 1, 60, 67, 57, 64, 50, 0.00],
#     ["Титов", 1, 1, 75, 98, 86, 82, 85, 1.50],
#     ["Чернова", 0, 1, 85, 85, 81, 85, 72, 1.25],
#     ["Четкин", 1, 1, 80, 56, 50, 69, 50, 0.00],
#     ["Шевченко", 1, 0, 55, 60, 30, 8, 60, 0.00]
# ]

# # Извлекаем x1-x7 (входные данные), x8 (стипендия) — отдельно
# X = [student[1:8] for student in raw_data]  # только x1-x7
# stipend = [student[8] for student in raw_data]  # x8

# # Функция нормализации признаков к диапазону [0, 1]
# def normalize(data):
#     # Транспонируем для нормализации по колонкам
#     cols = list(zip(*data))
#     normalized_cols = []
#     for col in cols:
#         min_val = min(col) 
#         max_val = max(col)
#         if max_val - min_val == 0:
#             normalized_cols.append([0.0] * len(col))
#         else:
#             normalized_cols.append([(x - min_val) / (max_val - min_val) for x in col])
#     return list(map(list, zip(*normalized_cols)))  # обратно в строки

# # Нормализуем входные данные
# X_norm = normalize(X)

# # === ШАГ 2. Инициализация сети Кохонена ===

# input_len = 7  # 7 входных параметров
# output_neurons = 4  # 4 нейрона-кластера

# # Случайная инициализация весов: 4 нейрона по 7 весов
# random.seed(32)  # фиксирует рандом
# weights = [[random.random() for _ in range(input_len)] for _ in range(output_neurons)]

# # === ШАГ 3. Обучение ===

# def euclidean_dist(vec1, vec2):
#     return sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5

# # Обучение: 6 эпох, в каждой по 20 итераций (проходов по выборке)
# learning_rate = 0.3 # начальный коэффициент скорости обучения
# for epoch in range(6): # 6 эпох
#     for i in range(len(X_norm)):
#         x = X_norm[i]
#         # Шаг 1: ищем ближайший нейрон
#         distances = [euclidean_dist(x, w) for w in weights]
#         winner_idx = distances.index(min(distances))
#         # Шаг 2: обновляем веса победителя
#         for j in range(input_len):
#             weights[winner_idx][j] += learning_rate * (x[j] - weights[winner_idx][j])
#     learning_rate -= 0.05  # уменьшаем скорость обучения c каждой эпохой

# # === ШАГ 4. Кластеризация ===

# # Присваиваем каждому студенту кластер
# clusters = []
# for x in X_norm:
#     distances = [euclidean_dist(x, w) for w in weights]
#     winner_idx = distances.index(min(distances))
#     clusters.append(winner_idx)

# # === ШАГ 5. Анализ кластеров по стипендии (x8) ===

# # Считаем среднюю стипендию по каждому кластеру
# cluster_data = [[] for _ in range(output_neurons)]
# for i, cluster_id in enumerate(clusters):
#     cluster_data[cluster_id].append(stipend[i])

# print("Анализ кластеров по стипендии:")
# for i, data in enumerate(cluster_data):
#     if data:
#         avg_stipend = sum(data) / len(data)
#         print(f"Кластер {i}: {len(data)} студентов, средняя стипендия = {avg_stipend:.2f}")
#     else:
#         print(f"Кластер {i}: пуст")

# # Вывод кластеров студентов
# print("\nКластеры студентов:")
# for i, student in enumerate(raw_data):
#     print(f"{student[0]} → Кластер {clusters[i]}")



import random

# === ШАГ 1. Подготовка данных ===

# Список студентов (Фамилия, Пол, Все зачеты, История, Графика, Математика, Химия, Физика, Стипендия)
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

# Входные данные (x1-x7) и отдельный список стипендий (x8)
X = [student[1:8] for student in raw_data]
stipend = [student[8] for student in raw_data]

# Нормализация признаков к [0,1]
def normalize(data):
    cols = list(zip(*data))
    normalized = []
    for col in cols:
        min_val, max_val = min(col), max(col)
        if max_val - min_val == 0:
            normalized.append([0.0] * len(col))
        else:
            normalized.append([(x - min_val) / (max_val - min_val) for x in col])
    return list(map(list, zip(*normalized)))

X_norm = normalize(X)

# === ШАГ 2. Инициализация сети Кохонена ===

input_len = 7  # 7 входов
output_neurons = 4  # 4 выхода

random.seed(30)  # фиксируем случайность
weights = [[random.random() for _ in range(input_len)] for _ in range(output_neurons)]

# === ШАГ 3. Обучение ===

def euclidean_dist(vec1, vec2):
    return sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5

learning_rate = 0.3

for epoch in range(6):  # 6 эпох
    for _ in range(20):  # 20 случайных обучающих примеров за эпоху
        x = random.choice(X_norm)
        distances = [euclidean_dist(x, w) for w in weights]
        winner_idx = distances.index(min(distances))
        for j in range(input_len):
            weights[winner_idx][j] += learning_rate * (x[j] - weights[winner_idx][j])
    learning_rate -= 0.05  # уменьшаем скорость обучения

# === ШАГ 4. Кластеризация ===

clusters = []
for x in X_norm:
    distances = [euclidean_dist(x, w) for w in weights]
    winner_idx = distances.index(min(distances))
    clusters.append(winner_idx)

# === ШАГ 5. Анализ кластеров по стипендии ===

cluster_data = [[] for _ in range(output_neurons)]
for i, cluster_id in enumerate(clusters):
    cluster_data[cluster_id].append(stipend[i])

print("\nАнализ кластеров по стипендии:")
for i, data in enumerate(cluster_data):
    if data:
        avg_stipend = sum(data) / len(data)
        print(f"Кластер {i}: {len(data)} студентов, средняя стипендия = {avg_stipend:.2f}")
    else:
        print(f"Кластер {i}: пуст")

# Вывод студентов по кластерам
# print("\nРаспределение студентов по кластерам:")
# for cluster_id in range(output_neurons):
#     print(f"\nКластер {cluster_id}:")
#     for i, student in enumerate(raw_data):
#         if clusters[i] == cluster_id:
#             print(f"  {student[0]}")
# === Новый красивый вывод студентов по кластерам ===

# Функция для форматированного вывода строки студента
def format_student(index, student):
    gender = "М" if student[1] == 1 else "Ж"
    credits = "Да" if student[2] == 1 else "Нет"
    history, graphics, math, chemistry, physics = student[3], student[4], student[5], student[6], student[7]
    stipend_formatted = f"{student[8]:.2f}".replace('.', ',')
    return f"{index+1:<2} {student[0]:<12} {gender:<1} {credits:<3} {history:<3} {graphics:<3} {math:<3} {chemistry:<3} {physics:<3} {stipend_formatted}"

print("\nРаспределение студентов по кластерам:")

for cluster_id in range(output_neurons):
    print(f"\n=== Кластер {cluster_id} ===")
    # Заголовок таблицы
    print("№  Фамилия     П  Зач  Ист  Граф  Мат  Хим  Физ  Стип")
    for i, student in enumerate(raw_data):
        if clusters[i] == cluster_id:
            print(format_student(i, student))

