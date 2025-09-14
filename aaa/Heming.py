import os, glob
from haming_network import HammingNetwork, load_patterns_from_dir,  load_tests_from_dir

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    etalon_dir = os.path.join(here, 'patterns')
    test_dir   = os.path.join(here, 'tests')

    # Отладка: куда мы смотрим и что там лежит
    print("Ищем эталоны в:", etalon_dir)
    print("  Содержимое каталога:", os.listdir(etalon_dir))
    print("  TXT-файлы:", glob.glob(os.path.join(etalon_dir, '*.txt')))
    patterns = load_patterns_from_dir(etalon_dir)
    print(f"Загружено {len(patterns)} эталонов.\n")

    print("Ищем тесты в:", test_dir)
    print("  Содержимое каталога:", os.listdir(test_dir))
    print("  TXT-файлы:", glob.glob(os.path.join(test_dir, '*.txt')))
    tests = load_tests_from_dir(test_dir)
    print(f"Загружено {len(tests)} тестов.\n")

    if not patterns:
        raise RuntimeError("Никаких эталонов не найдено — проверьте путь и расширения файлов")
    if not tests:
        raise RuntimeError("Никаких тестов не найдено — проверьте путь и расширения файлов")

    net = HammingNetwork(patterns)
    correct = 0

    for filename, vect, true_label in tests:
        pred = net.recognize(vect)
        ok = (str(pred) == true_label)
        marker = "✅" if ok else f"❌ (правильно {true_label})"
        print(f"{filename}: предсказано {pred} {marker}")
        if ok:
            correct += 1

    # Выводим итоговую точность
    total = len(tests)
    accuracy = correct / total * 100
    print(f"\nТочность: {accuracy:.1f}% ({correct}/{total})")

if __name__ == '__main__':
    main()