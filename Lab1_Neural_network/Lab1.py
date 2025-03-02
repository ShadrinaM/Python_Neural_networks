# # ЗАДАНИЕ1 просто
# import numpy as np

# # Функция для нормализации вектора
# def normalize_vector(vector):
#     norm = np.linalg.norm(vector)
#     if norm == 0:
#         return vector
#     return vector / norm

# # Инициализация весов нейронов случайными значениями и их нормализация
# def initialize_weights(num_neurons, input_size):
#     weights = np.random.rand(num_neurons, input_size)
#     for i in range(num_neurons):
#         weights[i] = normalize_vector(weights[i])
#     return weights

# # Обучение нейронов WTA
# def train_wta(weights, inputs, learning_rate=0.5, epochs=1):
#     for epoch in range(epochs):
#         for x in inputs:
#             # Вычисление выходных сигналов нейронов
#             outputs = np.dot(weights, x)
            
#             # Определение нейрона-победителя
#             winner = np.argmax(outputs)
            
#             # Обновление весов победившего нейрона
#             weights[winner] += learning_rate * (x - weights[winner])
            
#             # Нормализация весов победившего нейрона
#             weights[winner] = normalize_vector(weights[winner])
    
#     return weights

# # Входные данные (уже нормализованные)
# inputs = np.array([
#     [0.97, 0.20],
#     [1.00, 0.00],
#     [-0.72, 0.70],
#     [-0.67, 0.74],
#     [-0.80, 0.60],
#     [0.00, -1.00],
#     [0.20, -0.97],
#     [-0.30, -0.95]
# ])

# # Инициализация весов
# num_neurons = 4
# input_size = 2
# weights = initialize_weights(num_neurons, input_size)

# # Обучение нейронов
# learning_rate = 0.5
# trained_weights = train_wta(weights, inputs, learning_rate)

# # Вывод итоговых весов
# print("Веса нейронов после обучения:")
# for i, w in enumerate(trained_weights):
#     print(f"Нейрон {i+1}: {w}")

















# # ЗАДАНИЕ 1 С ПОДРОБНОСТЯМИ
# import numpy as np

# # Функция для нормализации вектора
# def normalize_vector(vector):
#     norm = np.linalg.norm(vector)
#     if norm == 0:
#         return vector
#     return vector / norm

# # Инициализация весов нейронов случайными значениями и их нормализация
# def initialize_weights(num_neurons, input_size):
#     weights = np.random.rand(num_neurons, input_size)
#     for i in range(num_neurons):
#         weights[i] = normalize_vector(weights[i])
#     return weights

# # Обучение нейронов WTA
# def train_wta(weights, inputs, learning_rate=0.5, epochs=1):
#     for epoch in range(epochs):
#         print(f"\nЭпоха {epoch + 1}:")
#         for i, x in enumerate(inputs):
#             print(f"\nОбработка входного вектора {i + 1}: {x}")
            
#             # Вычисление выходных сигналов нейронов
#             outputs = np.dot(weights, x)
            
#             # Определение нейрона-победителя
#             winner = np.argmax(outputs)
#             print(f"Нейрон-победитель: {winner + 1}")
            
#             # Обновление весов победившего нейрона
#             weights[winner] += learning_rate * (x - weights[winner])
            
#             # Нормализация весов победившего нейрона
#             weights[winner] = normalize_vector(weights[winner])
            
#             # Вывод весов после обновления
#             print("Веса нейронов после обновления:")
#             for j, w in enumerate(weights):
#                 print(f"Нейрон {j + 1}: {w}")
    
#     return weights

# # Входные данные (уже нормализованные)
# inputs = np.array([
#     [0.97, 0.20],
#     [1.00, 0.00],
#     [-0.72, 0.70],
#     [-0.67, 0.74],
#     [-0.80, 0.60],
#     [0.00, -1.00],
#     [0.20, -0.97],
#     [-0.30, -0.95]
# ])

# # Инициализация весов
# num_neurons = 4
# input_size = 2
# weights = initialize_weights(num_neurons, input_size)

# # Вывод начальных весов
# print("\nНачальные веса нейронов:")
# for i, w in enumerate(weights):
#     print(f"Нейрон {i + 1}: {w}")

# # Обучение нейронов
# learning_rate = 0.5
# trained_weights = train_wta(weights, inputs, learning_rate)

# # Вывод итоговых весов
# print("\nИтоговые веса нейронов после обучения:")
# for i, w in enumerate(trained_weights):
#     print(f"Нейрон {i + 1}: {w}")




























# # # ЗАДАНИЕ 2 БЕЗ ПОДРОБНОСТЕЙ

# # import numpy as np

# # # Функция для нормализации вектора
# # def normalize_vector(vector):
# #     norm = np.linalg.norm(vector)
# #     if norm == 0:
# #         return vector
# #     return vector / norm

# # # Инициализация весов нейронов случайными значениями и их нормализация
# # def initialize_weights(num_neurons, input_size):
# #     weights = np.random.rand(num_neurons, input_size)
# #     for i in range(num_neurons):
# #         weights[i] = normalize_vector(weights[i])
# #     return weights

# # # Модифицированное обучение WTA с учетом штрафов
# # def train_wta_modified(weights, inputs, learning_rate=0.5, epochs=1, penalty_factor=0.1):
# #     num_neurons = weights.shape[0]
# #     win_counts = np.zeros(num_neurons)  # Счетчик побед каждого нейрона
    
# #     for epoch in range(epochs):
# #         print(f"Эпоха {epoch + 1}:")
# #         for i, x in enumerate(inputs):
# #             print(f"\nОбработка входного вектора {i + 1}: {x}")
            
# #             # Вычисление выходных сигналов нейронов с учетом штрафов
# #             outputs = np.dot(weights, x)
# #             outputs = outputs / (1 + penalty_factor * win_counts)  # Штраф за частые победы
            
# #             # Определение нейрона-победителя
# #             winner = np.argmax(outputs)
# #             win_counts[winner] += 1  # Увеличиваем счетчик побед
# #             print(f"Нейрон-победитель: {winner + 1} (побед: {win_counts[winner]})")
            
# #             # Обновление весов победившего нейрона
# #             weights[winner] += learning_rate * (x - weights[winner])
            
# #             # Нормализация весов победившего нейрона
# #             weights[winner] = normalize_vector(weights[winner])
            
# #             # Вывод весов после обновления
# #             print("Веса нейронов после обновления:")
# #             for j, w in enumerate(weights):
# #                 print(f"Нейрон {j + 1}: {w} (побед: {win_counts[j]})")
    
# #     return weights

# # # Входные данные (уже нормализованные)
# # inputs = np.array([
# #     [0.97, 0.20],
# #     [1.00, 0.00],
# #     [-0.72, 0.70],
# #     [-0.67, 0.74],
# #     [-0.80, 0.60],
# #     [0.00, -1.00],
# #     [0.20, -0.97],
# #     [-0.30, -0.95]
# # ])

# # # Инициализация весов
# # num_neurons = 4
# # input_size = 2
# # weights = initialize_weights(num_neurons, input_size)

# # # Вывод начальных весов
# # print("\nНачальные веса нейронов:")
# # for i, w in enumerate(weights):
# #     print(f"Нейрон {i + 1}: {w}")

# # # Обучение нейронов с штрафом
# # learning_rate = 0.5  # Скорость обучения
# # penalty_factor = 0.1  # Коэффициент штрафа
# # trained_weights = train_wta_modified(weights, inputs, learning_rate, penalty_factor=penalty_factor)

# # # Вывод итоговых весов
# # print("\nИтоговые веса нейронов после обучения:")
# # for i, w in enumerate(trained_weights):
# #     print(f"Нейрон {i + 1}: {w}")



















# # ЗАДАНИЕ 2 С ПОДРОБНОСТЯМИ
# import numpy as np

# # Функция для нормализации вектора
# def normalize_vector(vector):
#     norm = np.linalg.norm(vector)
#     if norm == 0:
#         return vector
#     return vector / norm

# # Инициализация весов нейронов случайными значениями и их нормализация
# def initialize_weights(num_neurons, input_size):
#     weights = np.random.rand(num_neurons, input_size)
#     for i in range(num_neurons):
#         weights[i] = normalize_vector(weights[i])
#     return weights

# # Модифицированное обучение WTA с учетом штрафов
# def train_wta_modified(weights, inputs, learning_rate=0.5, epochs=1, penalty_factor=0.1):
#     num_neurons = weights.shape[0]
#     win_counts = np.zeros(num_neurons)  # Счетчик побед каждого нейрона
    
#     for epoch in range(epochs):
#         print(f"\nЭпоха {epoch + 1}:")
#         for i, x in enumerate(inputs):
#             print(f"\nОбработка входного вектора {i + 1}: {x}")
            
#             # Вычисление выходных сигналов нейронов без штрафа
#             outputs = np.dot(weights, x)
#             print("Выходные сигналы до штрафа:")
#             for j, output in enumerate(outputs):
#                 print(f"Нейрон {j + 1}: {output:.4f}")
            
#             # Применение штрафа к выходным сигналам
#             penalized_outputs = outputs / (1 + penalty_factor * win_counts)
#             print("Выходные сигналы после штрафа:")
#             for j, output in enumerate(penalized_outputs):
#                 print(f"Нейрон {j + 1}: {output:.4f} (побед: {win_counts[j]})")
            
#             # Определение нейрона-победителя
#             winner = np.argmax(penalized_outputs)
#             win_counts[winner] += 1  # Увеличиваем счетчик побед
#             print(f"Нейрон-победитель: {winner + 1} (побед: {win_counts[winner]})")
            
#             # Обновление весов победившего нейрона
#             weights[winner] += learning_rate * (x - weights[winner])
            
#             # Нормализация весов победившего нейрона
#             weights[winner] = normalize_vector(weights[winner])
            
#             # Вывод весов после обновления
#             print("Веса нейронов после обновления:")
#             for j, w in enumerate(weights):
#                 print(f"Нейрон {j + 1}: {w} (побед: {win_counts[j]})")
    
#     return weights

# # Входные данные (уже нормализованные)
# inputs = np.array([
#     [0.97, 0.20],
#     [1.00, 0.00],
#     [-0.72, 0.70],
#     [-0.67, 0.74],
#     [-0.80, 0.60],
#     [0.00, -1.00],
#     [0.20, -0.97],
#     [-0.30, -0.95]
# ])

# # Инициализация весов
# num_neurons = 4
# input_size = 2
# weights = initialize_weights(num_neurons, input_size)

# # Вывод начальных весов
# print("\nНачальные веса нейронов:")
# for i, w in enumerate(weights):
#     print(f"Нейрон {i + 1}: {w}")

# # Обучение нейронов с модификацией
# learning_rate = 0.5 # Скорость обучения
# penalty_factor = 0.1  # Коэффициент штрафа
# trained_weights = train_wta_modified(weights, inputs, learning_rate, penalty_factor=penalty_factor)

# # Вывод итоговых весов
# print("\nИтоговые веса нейронов после обучения:")
# for i, w in enumerate(trained_weights):
#     print(f"Нейрон {i + 1}: {w}")

























# # ЗАДАНИЕ 3 С ПОДРОБНОСТЯМИ
# import numpy as np

# # Функция для нормализации вектора
# def normalize_vector(vector):
#     norm = np.linalg.norm(vector)
#     if norm == 0:
#         return vector
#     return vector / norm

# # Инициализация весов случайными значениями
# def initialize_weights(num_neurons, input_size):
#     weights = np.random.rand(num_neurons, input_size)
#     for i in range(num_neurons):
#         weights[i] = normalize_vector(weights[i])
#     return weights

# # Функция активации (пороговая функция)
# def activation_function(x):
#     return 1 if x > 0 else 0

# # Обучение по правилу Хебба
# def train_hebb(weights, inputs, learning_rate=0.1, epochs=1):
#     num_neurons = weights.shape[0]
    
#     for epoch in range(epochs):
#         print(f"\nЭпоха {epoch + 1}:")
#         for i, x in enumerate(inputs):
#             print(f"\nОбработка входного вектора {i + 1}: {x}")
            
#             # Вычисление выходных сигналов нейронов
#             outputs = np.dot(weights, x)
#             y = np.array([activation_function(output) for output in outputs])
#             print(f"Выходные сигналы: {y}")
            
#             # Обновление весов по правилу Хебба
#             for j in range(num_neurons):
#                 weights[j] += learning_rate * y[j] * x
            
#             # # Нормализация весов
#             # for j in range(num_neurons):
#             #     weights[j] = normalize_vector(weights[j])
            
#             # Вывод весов после обновления
#             print("Веса нейронов после обновления:")
#             for j, w in enumerate(weights):
#                 print(f"Нейрон {j + 1}: {w}")
    
#     return weights

# # Входные данные (уже нормализованные)
# inputs = np.array([
#     [0.97, 0.20],
#     [1.00, 0.00],
#     [-0.72, 0.70],
#     [-0.67, 0.74],
#     [-0.80, 0.60],
#     [0.00, -1.00],
#     [0.20, -0.97],
#     [-0.30, -0.95]
# ])

# # Инициализация весов
# num_neurons = 2
# input_size = 2
# weights = initialize_weights(num_neurons, input_size)

# # Вывод начальных весов
# print("\nНачальные веса нейронов:")
# for i, w in enumerate(weights):
#     print(f"Нейрон {i + 1}: {w}")

# # Обучение по правилу Хебба
# learning_rate = 0.5 # Скорость обучения
# trained_weights = train_hebb(weights, inputs, learning_rate)

# # Вывод итоговых весов
# print("\nИтоговые веса нейронов после обучения:")
# for i, w in enumerate(trained_weights):
#     print(f"Нейрон {i + 1}: {w}")