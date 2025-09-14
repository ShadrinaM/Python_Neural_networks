import os
import random
import glob

def load_pattern(path):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    # объединяем в один список символов
    return list(''.join(lines))

def save_pattern(chars, path):
    # разбиваем обратно на 7 строк
    with open(path, 'w') as f:
        for i in range(0, 49, 7):
            f.write(''.join(chars[i:i+7]) + '\n')

def generate_noisy_variants(input_dir, output_dir, noise_levels, variants_per_level):
    os.makedirs(output_dir, exist_ok=True)
    for filepath in glob.glob(os.path.join(input_dir, '*.txt')):
        basename = os.path.splitext(os.path.basename(filepath))[0]  # '0', '1', ...
        orig = load_pattern(filepath)
        for noise in noise_levels:
            flips = int(49 * noise / 100)
            for v in range(1, variants_per_level + 1):
                noisy = orig.copy()
                idxs = random.sample(range(49), flips)
                for i in idxs:
                    noisy[i] = '0' if noisy[i] == '1' else '1'
                out_name = f"{basename}_noise{noise:02d}_{v:02d}.txt"
                save_pattern(noisy, os.path.join(output_dir, out_name))
    print("Генерация завершена.")

if __name__ == '__main__':
    # параметры: уровни шума в процентах и количество вариантов на уровень
    noise_levels = [5, 10, 20]
    variants_per_level = 5
    generate_noisy_variants(
        input_dir='patterns',
        output_dir='tests',
        noise_levels=noise_levels,
        variants_per_level=variants_per_level
    )
