import os
import glob
import math
import random

# Загрузка зашумлённых образов вместе с именами файлов.
def load_tests_from_dir(dir_path):
    tests = []
    files = sorted(glob.glob(os.path.join(dir_path, '*.txt')))
    for filepath in files:
        filename = os.path.basename(filepath)
        # Истинный класс — всё до первого подчёркивания, либо до точки
        true_label = filename.split('_')[0]
        # Загружаем вектор как раньше
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        vect = []
        for line in lines:
            for ch in line:
                if ch == '1':
                    vect.append(1)
                elif ch == '0':
                    vect.append(-1)
                else:
                    raise ValueError(f"Недопустимый символ '{ch}' в {filename}")
        tests.append((filename, vect, true_label))
    return tests

# Рекуррентная сеть Хэмминга для распознавания образов.
class HammingNetwork:
    # Инициализация сети.
    def __init__(self, patterns, epsilon=1e-3):
        self.patterns = patterns
        self.p = len(patterns)      # число классов
        if self.p == 0:
            raise ValueError("Нужно как минимум один эталонный образ.")
        self.N = len(patterns[0])    # размер входа
        # Проверка одинаковой размерности образов
        for pat in patterns:
            if len(pat) != self.N:
                raise ValueError("Все эталоны должны иметь одинаковую длину.")

        # Формируем матрицу весов первого слоя W1[j][i] = patterns[j][i]
        self.W1 = [list(pat) for pat in patterns]

        # Формируем матрицу латеральных весов MaxNet Wm[j][k]
        self.Wm = []
        for j in range(self.p):
            row = []
            for k in range(self.p):
                if j == k:
                    row.append(1.0)
                else:
                    noise = (random.random() - 0.5) * epsilon
                    row.append(-1.0 / (self.p - 1) + noise)
            self.Wm.append(row)

    @staticmethod
    def activation(y):
        """Линейная пороговая функция: f(y) = y, если y >= 0; иначе 0"""
        return y if y >= 0 else 0

# Распознавание образа x.
    def recognize(self, x, E_max=0.1, max_iter=100):
        # Слой 1: вычисляем выходы как скалярные произведения
        y1 = []
        for j in range(self.p):
            s = sum(self.W1[j][i] * x[i] for i in range(self.N))
            # Пороговая активация
            y1.append(self.activation(s))

        # Инициализируем MaxNet
        y = y1.copy()
        for _ in range(max_iter):
            y_prev = y.copy()
            # Обновление каждого выходного нейрона
            for j in range(self.p):
                total = sum(self.Wm[j][k] * y_prev[k] for k in range(self.p))
                y[j] = self.activation(total)
            # Проверка стабилизации
            diff = math.sqrt(sum((y[j] - y_prev[j])**2 for j in range(self.p)))
            if diff <= E_max:
                break

        # Определяем победителя
        positives = [j for j, val in enumerate(y) if val > 0]
        return positives[0] if len(positives) == 1 else None

#Загрузка образов из текстовых файлов в папке.
def load_patterns_from_dir(dir_path):
    patterns = []
    files = sorted(glob.glob(os.path.join(dir_path, '*.txt')))
    for filepath in files:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        vect = []
        for line in lines:
            for ch in line:
                if ch == '1': vect.append(1)
                elif ch == '0': vect.append(-1)
                else:
                    raise ValueError(f"Недопустимый символ '{ch}' в файле {filepath}")
        patterns.append(vect)
    return patterns