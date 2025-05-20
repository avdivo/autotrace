import cv2
import numpy as np

# Исходная
# def compute_dhash_vector(image: np.ndarray) -> np.ndarray:
#     """
#     Вычисляет dHash региона в виде бинарного вектора (64 элемента: 0 или 1).
#
#     Args:
#         image (np.ndarray): Полное изображение (формат BGR).
#
#     Returns:
#         np.ndarray: Вектор хэша (shape: (64,), dtype: uint8).
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Преобразуем в градации серого
#     resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)  # Сжимаем до 9x8
#     diff = resized[:, 1:] > resized[:, :-1]  # Побитовое сравнение по строкам
#     vec = diff.flatten().astype(np.uint8)  # Преобразуем в одномерный массив (64 бита)
#     return vec
# Новая
def compute_dhash_vector(image: np.ndarray) -> np.ndarray:
    """
    Вычисляет pHash региона в виде бинарного вектора (64 элемента: 0 или 1).
    (Реализация использует pHash, но сохраняет интерфейс как у dHash)

    Args:
        image (np.ndarray): Полное изображение (формат BGR).

    Returns:
        np.ndarray: Вектор хэша (shape: (64,), dtype: uint8).
    """
    # Преобразование в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Уменьшение размера до 32x32 (стандартный размер для pHash)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)

    # Преобразование в частотную область с помощью DCT
    dct = cv2.dct(np.float32(resized))

    # Берем верхние 8x8 коэффициентов (исключая DC-компоненту)
    dct_roi = dct[:8, 1:9]

    # Вычисляем среднее значение (исключая первый коэффициент)
    avg = np.mean(dct_roi)

    # Создаем бинарный вектор (1 если значение > среднего)
    diff = dct_roi > avg
    vec = diff.flatten().astype(np.uint8)

    return vec

def dhash_vector_to_hex(bits: np.ndarray) -> str:
    """
    Преобразует бинарный вектор хэша в шестнадцатеричную строку.

    Args:
        bits (np.ndarray): Бинарный вектор dHash (длина 64).

    Returns:
        str: Hex-строка длиной 16 символов.
    """
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)  # Сдвигаем и добавляем бит
    return f"{value:016x}"  # Преобразуем в hex с ведущими нулями


def hex_to_dhash_vector(hex_str: str) -> np.ndarray:
    """
    Преобразует шестнадцатеричную строку хэша обратно в бинарный вектор.

    Args:
        hex_str (str): Хэш в hex-формате (16 символов).

    Returns:
        np.ndarray: Бинарный вектор dHash (shape: (64,), dtype: uint8).
    """
    value = int(hex_str, 16)
    bits = [(value >> (63 - i)) & 1 for i in range(64)]  # Восстановление битов в правильном порядке
    return np.array(bits, dtype=np.uint8)

# Начальный вариант сравнения
# def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
#     """
#     Вычисляет косинусное сходство между двумя векторами.
#
#     Косинусное сходство показывает степень схожести между двумя векторами,
#     измеряя угол между ними. Значения находятся в диапазоне [0, 1]:
#       - `1.0` → Полное совпадение (вектора идентичны)
#       - `0.0` → Нет сходства (ортогональные вектора)
#       - `-1.0` → Полная противоположность (вектора направлены в разные стороны)
#
#     Если один или оба вектора пустые (`норма = 0`), функция возвращает:
#       - `1.0`, если оба вектора пустые (они идентичны)
#       - `0.0`, если один из векторов пуст, а другой нет
#
#     Args:
#         vec1 (np.ndarray): Первый вектор (обычно 64-битный dHash).
#         vec2 (np.ndarray): Второй вектор для сравнения.
#
#     Returns:
#         float: Косинусное сходство между `vec1` и `vec2` в диапазоне [-1, 1].
#
#     Raises:
#         ValueError: Если длины `vec1` и `vec2` не совпадают.
#     """
#     if vec1.shape != vec2.shape:
#         raise ValueError(f"Несовпадающие размеры векторов: {vec1.shape} vs {vec2.shape}")
#
#     dot_product = np.dot(vec1, vec2)  # Скалярное произведение
#     norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)  # Вычисление норм
#
#     if norm1 == 0 and norm2 == 0:
#         return 1.0  # Полное совпадение пустых векторов
#     if norm1 == 0 or norm2 == 0:
#         return 0.0  # Если один из векторов пуст, а другой нет, сходство отсутствует
#
#     norm_product = norm1 * norm2
#     return dot_product / norm_product if norm_product > 0 else 0.0


# Альтернативные варианты сравнения
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Сравнивает два бинарных хэша (pHash/dHash) через расстояние Хэмминга.
    Возвращает нормализованную схожесть [0, 1], где:
      - `1.0` — полное совпадение (расстояние = 0)
      - `0.0` — максимальное несовпадение (все биты разные)

    Оптимизировано для сравнения небольших UI-элементов (кнопки, иконки).

    Args:
        vec1 (np.ndarray): Бинарный вектор (64 бита, dtype=uint8).
        vec2 (np.ndarray): Бинарный вектор для сравнения.

    Returns:
        float: Схожесть в диапазоне [0, 1].

    Raises:
        ValueError: Если длины векторов не равны 64.
    """
    if len(vec1) != 64 or len(vec2) != 64:
        raise ValueError("Векторы должны быть длины 64")

    # Расстояние Хэмминга (количество разных битов)
    hamming_dist = np.sum(vec1 != vec2)

    # Нормализация до [0, 1] (1 - полное совпадение)
    similarity = 1.0 - (hamming_dist / 64)
    return similarity


