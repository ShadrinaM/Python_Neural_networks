import os
import glob
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Сеть Хэмминга для распознавания образов.
class HammingNetwork:
    # Инициализация сети Хэмминга.
    def __init__(self, patterns, epsilon=1e-3):
       
        # Преобразование 2D образцов в 1D векторы
        self.patterns = [self._convert_to_1d(p) for p in patterns]
        self.p = len(self.patterns)  # Количество эталонных образов
        if self.p == 0:
            raise ValueError("Необходимо хотя бы один эталонный образ")
            
        self.N = len(self.patterns[0])  # Размерность входного вектора
        
        # Проверка одинаковой размерности всех образцов
        for pat in self.patterns:
            if len(pat) != self.N:
                raise ValueError("Все образцы должны иметь одинаковую размерность")
        
        # Матрица весов первого слоя (эталонные образцы)
        self.W1 = [list(pat) for pat in self.patterns]
        
        # Матрица весов MaxNet (латеральное торможение)
        self.Wm = []
        for j in range(self.p):
            row = []
            for k in range(self.p):
                if j == k:
                    row.append(1.0)  # Положительная обратная связь
                else:
                    # Отрицательная обратная связь с небольшим шумом
                    noise = (random.random() - 0.5) * epsilon
                    row.append(-1.0 / (self.p - 1) + noise)
            self.Wm.append(row)
    
    def _convert_to_1d(self, pattern_2d):
        """Преобразует 2D массив в 1D вектор"""
        return np.array(pattern_2d).flatten()
    
    def _convert_to_2d(self, pattern_1d, shape):
        """Преобразует 1D вектор обратно в 2D массив"""
        return np.array(pattern_1d).reshape(shape)
    
    @staticmethod
    def activation(y):
        """Функция активации: f(y) = y, если y >= 0, иначе 0"""
        return y if y >= 0 else 0
    
    # Распознавание входного образца.
    def recognize(self, x, E_max=0.1, max_iter=100):
        # Преобразование входного образца в 1D
        x_1d = self._convert_to_1d(x)
        
        # Первый слой: вычисление сходства с эталонами
        y1 = []
        for j in range(self.p):
            s = sum(self.W1[j][i] * x_1d[i] for i in range(self.N))
            y1.append(self.activation(s))
        
        # Второй слой: конкурентное взаимодействие (MaxNet)
        y = y1.copy()
        for _ in range(max_iter):
            y_prev = y.copy()
            for j in range(self.p):
                total = sum(self.Wm[j][k] * y_prev[k] for k in range(self.p))
                y[j] = self.activation(total)
            
            # Проверка стабилизации
            diff = math.sqrt(sum((y[j] - y_prev[j])**2 for j in range(self.p)))
            if diff <= E_max:
                break
        
        # Определение победителя
        positives = [j for j, val in enumerate(y) if val > 0]
        return positives[0] if len(positives) == 1 else None

# Загрузка эталонных образов из директории.
def load_patterns_from_dir(dir_path, shape=(7,7)):
    patterns = []
    for filepath in sorted(glob.glob(os.path.join(dir_path, '*.txt'))):
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Создаем 2D массив
        pattern = []
        for line in lines:
            row = []
            for ch in line:
                if ch == '1':
                    row.append(1)
                elif ch == '0':
                    row.append(-1)
                else:
                    raise ValueError(f"Недопустимый символ '{ch}' в файле {filepath}")
            pattern.append(row)
        
        # Проверка размерности
        if len(pattern) != shape[0] or len(pattern[0]) != shape[1]:
            raise ValueError(f"Неверная размерность образца в файле {filepath}")
        
        patterns.append(pattern)
    return patterns

#  Загрузка тестовых образов из директории.
def load_tests_from_dir(dir_path, shape=(7,7)):
    tests = []
    for filepath in sorted(glob.glob(os.path.join(dir_path, '*.txt'))):
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        true_label = parts[0]  # Извлекаем метку класса
        
        # Извлекаем уровень шума из имени файла
        noise_level = 0
        for part in parts:
            if part.startswith('noise'):
                noise_level = int(part[5:])
                break
        
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Создаем 2D массив
        pattern = []
        for line in lines:
            row = []
            for ch in line:
                if ch == '1':
                    row.append(1)
                elif ch == '0':
                    row.append(-1)
                else:
                    raise ValueError(f"Недопустимый символ '{ch}' в файле {filename}")
            pattern.append(row)
        
        # Проверка размерности
        if len(pattern) != shape[0] or len(pattern[0]) != shape[1]:
            raise ValueError(f"Неверная размерность теста в файле {filename}")
        
        tests.append((filename, pattern, true_label, noise_level))
    return tests

# Визуализация образца.
def visualize_pattern(pattern, title="", size=(7,7)):
    plt.imshow(pattern, cmap='binary', vmin=-1, vmax=1)
    plt.title(title)
    plt.colorbar()
    plt.show()

# Генерация зашумленных вариантов эталонных образов.
def generate_noisy_variants(input_dir, output_dir, noise_levels, variants_per_level, shape=(7,7)):
    os.makedirs(output_dir, exist_ok=True)
    
    for filepath in glob.glob(os.path.join(input_dir, '*.txt')):
        basename = os.path.splitext(os.path.basename(filepath))[0]
        pattern = load_patterns_from_dir(input_dir, shape)[0]  # Загружаем как 2D
        
        for noise in noise_levels:
            for v in range(1, variants_per_level + 1):
                # Создаем зашумленную копию
                noisy_pattern = [row.copy() for row in pattern]
                total_pixels = shape[0] * shape[1]
                flips = int(total_pixels * noise / 100)
                
                # Инвертируем случайные пиксели
                for _ in range(flips):
                    i, j = random.randint(0, shape[0]-1), random.randint(0, shape[1]-1)
                    noisy_pattern[i][j] *= -1
                
                # Сохраняем в файл
                out_name = f"{basename}_noise{noise:02d}_{v:02d}.txt"
                with open(os.path.join(output_dir, out_name), 'w') as f:
                    for row in noisy_pattern:
                        line = ''.join(['1' if x == 1 else '0' for x in row])
                        f.write(line + '\n')
    
    print(f"Сгенерировано зашумленных образцов в {output_dir}")

# Выводит результаты распознавания по уровням шума.
def print_results_by_noise_level(results):
    noise_levels = sorted({nl for digit in results for nl in results[digit]})
    
    print("\nРезультаты распознавания по уровням шума:")
    print("Цифра | " + " | ".join(f"{nl}% шума" for nl in noise_levels))
    print("-" * (6 + len(noise_levels) * 10))
    
    for digit in sorted(results.keys()):
        row = [f"{digit:^5}"]
        for noise in noise_levels:
            if noise in results[digit]:
                correct, total = results[digit][noise]
                accuracy = correct / total * 100 if total > 0 else 0
                row.append(f"{accuracy:>6.1f}%")
            else:
                row.append(" " * 6)
        print(" | ".join(row))

def main():
    # Пути к директориям
    here = os.path.dirname(os.path.abspath(__file__))
    etalon_dir = os.path.join(here, 'patterns')
    test_dir = os.path.join(here, 'tests')
    
    # Загрузка эталонных образцов
    print(f"Загрузка эталонов из {etalon_dir}")
    patterns = load_patterns_from_dir(etalon_dir)
    print(f"Загружено {len(patterns)} эталонов")
    
    # Визуализация эталонов
    for i, pattern in enumerate(patterns):
        visualize_pattern(pattern, title=f"Эталонный образец {i+1}")
    
    # Загрузка тестовых образцов
    print(f"\nЗагрузка тестов из {test_dir}")
    tests = load_tests_from_dir(test_dir)
    print(f"Загружено {len(tests)} тестов")
    
    # Создание и тестирование сети
    net = HammingNetwork(patterns)
    
    # Словарь для хранения результатов: {digit: {noise_level: (correct, total)}}
    results = {}
    
    print("\nТестирование сети:")
    for filename, pattern, true_label, noise_level in tests:
        pred = net.recognize(pattern)
        ok = (str(pred) == true_label)
        marker = "✅" if ok else f"❌ (правильно {true_label})"
        print(f"{filename}: шум {noise_level}%, предсказано {pred} {marker}")
        
        # Обновляем статистику
        if true_label not in results:
            results[true_label] = {}
        if noise_level not in results[true_label]:
            results[true_label][noise_level] = (0, 0)
        
        correct, total = results[true_label][noise_level]
        results[true_label][noise_level] = (correct + (1 if ok else 0), total + 1)
    
    # Вывод результатов по уровням шума для каждой цифры
    print_results_by_noise_level(results)

if __name__ == '__main__':    
    # Запуск основной программы
    main()