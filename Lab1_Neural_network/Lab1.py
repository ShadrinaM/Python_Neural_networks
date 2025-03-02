# import numpy as np

# # Входные векторы
# X = np.array([
#     [0.97, 0.20],
#     [1.00, 0.00],
#     [-0.72, 0.70],
#     [-0.67, 0.74],
#     [-0.80, 0.60],
#     [0.00, -1.00],
#     [0.20, -0.97],
#     [-0.30, -0.95]
# ])

# # Количество нейронов
# num_neurons = 4

# # Инициализация весов случайным образом и нормализация
# np.random.seed(42)  # Для воспроизводимости результатов
# weights = np.random.rand(num_neurons, 2)
# weights /= np.linalg.norm(weights, axis=1, keepdims=True)

# # Коэффициент обучения
# eta = 0.5

# # Обучение сети
# for x in X:
#     # Вычисление выходных сигналов нейронов
#     u = np.dot(weights, x)
    
#     # Определение нейрона-победителя
#     winner = np.argmax(u)
    
#     # Обновление весов победившего нейрона по правилу Гроссберга
#     weights[winner] += eta * (x - weights[winner])
    
#     # Нормализация весов победившего нейрона
#     weights[winner] /= np.linalg.norm(weights[winner])

# # Вывод весов после обучения
# print("Веса нейронов после обучения:")
# for i, w in enumerate(weights):
#     print(f"Нейрон {i+1}: {w}")




import numpy as np

# Функция для нормализации вектора
def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

# Инициализация весов нейронов случайными значениями и их нормализация
def initialize_weights(num_neurons, input_size):
    weights = np.random.rand(num_neurons, input_size)
    for i in range(num_neurons):
        weights[i] = normalize_vector(weights[i])
    return weights

# Обучение нейронов WTA
def train_wta(weights, inputs, learning_rate=0.5, epochs=1):
    for epoch in range(epochs):
        for x in inputs:
            # Вычисление выходных сигналов нейронов
            outputs = np.dot(weights, x)
            
            # Определение нейрона-победителя
            winner = np.argmax(outputs)
            
            # Обновление весов победившего нейрона
            weights[winner] += learning_rate * (x - weights[winner])
            
            # Нормализация весов победившего нейрона
            weights[winner] = normalize_vector(weights[winner])
    
    return weights

# Входные данные (уже нормализованные)
inputs = np.array([
    [0.97, 0.20],
    [1.00, 0.00],
    [-0.72, 0.70],
    [-0.67, 0.74],
    [-0.80, 0.60],
    [0.00, -1.00],
    [0.20, -0.97],
    [-0.30, -0.95]
])

# Инициализация весов
num_neurons = 4
input_size = 2
weights = initialize_weights(num_neurons, input_size)

# Обучение нейронов
learning_rate = 0.5
trained_weights = train_wta(weights, inputs, learning_rate)

# Вывод итоговых весов
print("Веса нейронов после обучения:")
for i, w in enumerate(trained_weights):
    print(f"Нейрон {i+1}: {w}")
