import random

### ПОДГОТОВКА ДАННЫХ ###
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

# Нормализация признаков к [0,1] на основе максимума и минимума значений параметра
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


### ИНИЦИАЛИЗАЦИЯ СЕТИ КОХОНЕНА ###
input_len = 7  # 7 входа
output_neurons = 4  # 4 выхода

random.seed(47)  # фиксация случайности
weights = [[random.random() for _ in range(input_len)] for _ in range(output_neurons)]


### ОБУЧЕНИЕ ###
# Функция активации
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


### КЛАСТЕРИЗАЦИЯ ###
clusters = []
for x in X_norm:
    distances = [euclidean_dist(x, w) for w in weights]
    winner_idx = distances.index(min(distances))
    clusters.append(winner_idx)


### АНАЛИЗ КЛАСТЕРОВ ПО СТИПЕНДИИ ###
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


### ВЫВОД СТУДЕНТОВ ПО КЛАСТЕРАМ ###
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


### ОБРАБОТКА НОВОГО СТУДЕНТА ###
# Функция нормализации нового студента (с учетом min и max от обучающей выборки)
def normalize_single(new_data, original_data):
    normalized = []
    for idx, value in enumerate(new_data):
        col = [row[idx] for row in original_data]
        min_val, max_val = min(col), max(col)
        if max_val - min_val == 0:
            normalized.append(0.0)
        else:
            normalized.append((value - min_val) / (max_val - min_val))
    return normalized
# Запрос данных нового студента
print("\nВведите данные нового студента:")
new_gender = int(input("Пол (1 - М, 0 - Ж): "))
new_credits = int(input("Все зачеты получены (1 - Да, 0 - Нет): "))
new_history = int(input("Оценка по Истории: "))
new_graphics = int(input("Оценка по Инженерной графике: "))
new_math = int(input("Оценка по Математике: "))
new_chemistry = int(input("Оценка по Органической химии: "))
new_physics = int(input("Оценка по Физике: "))
# Подготовка вектора нового студента
new_student = [new_gender, new_credits, new_history, new_graphics, new_math, new_chemistry, new_physics]
# Нормализация нового студента
new_student_norm = normalize_single(new_student, X)
# Определение кластера
distances_new = [euclidean_dist(new_student_norm, w) for w in weights]
new_cluster = distances_new.index(min(distances_new))
print(f"\nНовый студент отнесен к кластеру: {new_cluster}")