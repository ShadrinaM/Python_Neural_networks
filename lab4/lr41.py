import numpy as np
import matplotlib.pyplot as plt
import random

class HopfieldNetwork:
    # Инициализация сети Хопфилда (принимает размер сети)
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))  # Матрица весовых коэффициентов
        
    # Обучение сети по правилу Хебба (принимает список образцов для обучения (каждый образец - 1D массив из -1 и 1))
    def train(self, patterns):
        num_patterns = len(patterns)
        n = self.size
        
        # Обнуление весов
        self.weights = np.zeros((n, n))
        
        # По правилу Хебба
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Суммируем произведения компонент всех образцов
                    for pattern in patterns:
                        self.weights[i, j] += pattern[i] * pattern[j]
                    # Нормирование
                    self.weights[i, j] /= n
        
        # Диагональные элементы нулевые (из себя в себя)
        np.fill_diagonal(self.weights, 0)
    
    # Распознавание образа (асинхронное обновление)
    def predict(self, input_pattern, max_iter=100):
        # Копируем входной образ, чтобы не изменять оригинал
        pattern = np.copy(input_pattern)
        n = self.size
        
        for iteration in range(max_iter):
            old_pattern = np.copy(pattern)
            
            # Асинхронное обновление (нейроны обновляются в случайном порядке)
            neurons_order = list(range(n))
            random.shuffle(neurons_order)
            
            for i in neurons_order:
                # Вычисляем новое состояние нейрона
                activation = np.dot(self.weights[i, :], pattern)
                pattern[i] = 1 if activation >= 0 else -1
            
            # Проверка на достижение устойчивого состояния
            if np.array_equal(pattern, old_pattern):
                return pattern, iteration + 1
        
        return pattern, max_iter

# Преобразование в одномерные векторы (принимает двумерный массив из -1 и 1, возвращает отдномерный)
def create_image_pattern(image_data):
    return image_data.flatten()

# Добавление шума к образцу (принимает идеальный образец одномерным вектором, уровень шума 0-1)
def add_noise(pattern, noise_level):
    noisy_pattern = np.copy(pattern)
    n = len(pattern)
    num_noisy = int(n * noise_level)
    
    # Выбираем случайные индексы для добавления шума
    noisy_indices = random.sample(range(n), num_noisy)
    
    for idx in noisy_indices:
        noisy_pattern[idx] *= -1  # Инвертируем значение
    
    return noisy_pattern

# Визуализация образца (принимает одномерный вектор соответствующий изображению, название и размер изображения)
def visualize_pattern(pattern, title="", size=(10, 10)):
    plt.imshow(pattern.reshape(size), cmap='binary', vmin=-1, vmax=1)
    plt.title(title)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    # Размер сети (10x10 изображение = число входов = 100 нейронов )
    N = 100
    hn = HopfieldNetwork(N)
    
    # Образцы для обучения
    # Образец 1: Буква "T"
    pattern1 = -np.ones((10, 10))  # Изначально все черное
    pattern1[0, :] = 1  # Первая строка белая
    pattern1[:, 4:6] = 1  # Вертикальная линия в середине
    
    # Образец 2: Буква "L"
    pattern2 = -np.ones((10, 10))
    pattern2[:, 0] = 1  # Первый столбец белый
    pattern2[-1, :] = 1  # Последняя строка белая
    
    # Преобразование в одномерные векторы
    pattern1_flat = create_image_pattern(pattern1)
    pattern2_flat = create_image_pattern(pattern2)
    
    # Обучение сети на основе двух идеальных образцов
    hn.train([pattern1_flat, pattern2_flat])
    
    # Тестирование сети с различными уровнями шума
    test_patterns = [pattern1_flat, pattern2_flat]
    pattern_names = ["T", "L"]    
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]  # Уровни шума от 10% до 50%
    
    for i, (pattern, name) in enumerate(zip(test_patterns, pattern_names)):
        print(f"\nТестирование образца '{name}':")
        
        for noise in noise_levels:
            # Добавляем шум
            noisy_pattern = add_noise(pattern, noise)
            
            # Визуализация зашумленного образца
            visualize_pattern(noisy_pattern, title=f"Зашумленный образец '{name}' (шум: {noise*100}%)")
            
            # Распознавание
            recognized, iterations = hn.predict(noisy_pattern)
            
            # Визуализация результата
            visualize_pattern(recognized, title=f"Распознанный образ (шум: {noise*100}%)")
            
            # Проверка, совпадает ли результат с одним из запомненных образцов
            if np.array_equal(recognized, pattern1_flat):
                print(f"При шуме {noise*100}%: распознано как 'T' за {iterations} итераций")
            elif np.array_equal(recognized, pattern2_flat):
                print(f"При шуме {noise*100}%: распознано как 'L' за {iterations} итераций")
            else:
                print(f"При шуме {noise*100}%: не удалось распознать образ (итераций: {iterations})")
            
