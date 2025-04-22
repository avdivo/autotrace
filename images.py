"""
Модуль для работы с изображениями и их векторными представлениями.

Модуль содержит функции для обновления текущего экрана программы,
сохранения области экрана с событием мыши и получения координат
указанного изображения на экране.
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional

import settings
from ml_model import model
from chroma_db import chroma_db
from manager import Manager
from exceptions import ScreenCaptureError, EmbeddingError


def screen_update(manager: Manager) -> str:
    """
    Обновляет текущий экран программы, получает его изображение и идентификатор.
    
    Функция захватывает снимок экрана, извлекает его часть для обработки,
    получает векторное представление и находит или создает идентификатор экрана.
    
    Args:
        manager: Объект Manager для доступа к компонентам системы
    
    Returns:
        str: Идентификатор текущего экрана (screen_id)
        
    Raises:
        ScreenCaptureError: Если не удалось получить снимок экрана
        EmbeddingError: Если не удалось получить векторное представление
    """
    # Получаем скриншот с помощью ScreenMonitor
    screenshot = manager.monitor.get_screenshot()
    if screenshot is None:
        # Если скриншот не получен, возвращаем текущий screen_id
        return manager.screen_id
    
    # Сохраняем скриншот в Manager
    manager.screenshot = screenshot
    
    # Получаем настройки для размера области и её расположения
    area_size = settings.SCREEN_AREA_SIZE
    area_x, area_y = settings.TOP_LEFT_CORNER
    
    # Извлекаем область верхнего левого угла (или другую, заданную в настройках)
    screen_area = screenshot[area_y:area_y+area_size[1], area_x:area_x+area_size[0]]
    
    # Получаем векторное представление выбранного участка
    try:
        vector = model.get_embedding(screen_area)
    except Exception as e:
        raise EmbeddingError(f"Ошибка при получении эмбеддинга экрана: {e}")
    
    # Проверяем наличие screen_id по вектору
    screen_id = chroma_db.get_screen_id(vector)
    
    # Если screen_id не найден, создаем новый
    if screen_id is None:
        screen_id = settings.generate_unique_id()
        chroma_db.create_screen(screen_id, vector)
        
    # Сохраняем полный скриншот в папке screenshots
    screenshots_folder = settings.SCREENSHOTS_DIR
    os.makedirs(screenshots_folder, exist_ok=True)
    screenshot_path = os.path.join(screenshots_folder, f"{screen_id}.png")
    cv2.imwrite(screenshot_path, screenshot)
    
    # Сохраняем screen_id в Manager и возвращаем его
    manager.screen_id = screen_id
    return screen_id


def set_sample(manager: Manager, x: int, y: int) -> str:
    """
    Сохраняет область экрана, в которой произошло событие мыши, и получает её идентификатор.
    
    Args:
        manager: Объект Manager для доступа к компонентам системы
        x: Координата X указателя мыши
        y: Координата Y указателя мыши
        
    Returns:
        str: Идентификатор образца (sample_id)
        
    Raises:
        ScreenCaptureError: Если текущий скриншот недоступен
        EmbeddingError: Если не удалось получить векторное представление
    """
    # Проверяем наличие скриншота
    if manager.screenshot is None:
        raise ScreenCaptureError("Текущий скриншот недоступен")
    
    # Получаем размер образца из настроек
    sample_size = settings.SAMPLE_AREA_SIZE
    
    # Вычисляем координаты для извлечения образца
    half_size_x, half_size_y = sample_size[0] // 2, sample_size[1] // 2
    x1 = max(0, x - half_size_x)
    y1 = max(0, y - half_size_y)
    x2 = min(manager.screenshot.shape[1], x + half_size_x)
    y2 = min(manager.screenshot.shape[0], y + half_size_y)
    
    # Извлекаем сегмент изображения
    sample_image = manager.screenshot[y1:y2, x1:x2]
    
    # Если размер извлеченного образца не соответствует требуемому, изменяем его
    if sample_image.shape[0] != sample_size[1] or sample_image.shape[1] != sample_size[0]:
        sample_image = cv2.resize(sample_image, sample_size)
    
    # Получаем векторное представление образца
    try:
        vector = model.get_embedding(sample_image)
    except Exception as e:
        raise EmbeddingError(f"Ошибка при получении эмбеддинга образца: {e}")
    
    # Создаем метаданные с текущим screen_id
    metadata = {'screen_id': manager.screen_id}
    
    # Проверяем наличие sample_id по вектору и метаданным
    sample_id = chroma_db.get_sample_id(vector, metadata)
    
    # Если sample_id не найден, создаем новый
    if sample_id is None:
        sample_id = settings.generate_unique_id()
        chroma_db.create_sample(sample_id, vector, metadata)
    
    # Сохраняем образец в папке sample
    samples_folder = settings.SAMPLES_DIR
    os.makedirs(samples_folder, exist_ok=True)
    sample_path = os.path.join(samples_folder, f"{sample_id}.png")
    cv2.imwrite(sample_path, sample_image)
    
    return sample_id


def get_xy(manager: Manager, sample_id: str, vector: np.ndarray) -> Tuple[int, int]:
    """
    Находит координаты на экране, где расположено изображение с указанным sample_id.
    
    Args:
        manager: Объект Manager для доступа к компонентам системы
        sample_id: Идентификатор образца
        vector: Векторное представление образца
        
    Returns:
        Tuple[int, int]: Координаты (x, y) найденного образца
        
    Raises:
        FileNotFoundError: Если файл образца не найден
        ScreenCaptureError: Если текущий скриншот недоступен
        ValueError: Если образец не найден на текущем экране
    """
    # Проверяем наличие скриншота
    if manager.screenshot is None:
        raise ScreenCaptureError("Текущий скриншот недоступен")
    
    # Формируем путь к файлу образца
    samples_folder = settings.SAMPLES_DIR
    sample_path = os.path.join(samples_folder, f"{sample_id}.png")
    
    # Проверяем существование файла образца
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Файл образца {sample_path} не найден")
    
    # Загружаем изображение образца
    sample_image = cv2.imread(sample_path)
    
    if sample_image is None:
        raise FileNotFoundError(f"Не удалось загрузить файл образца {sample_path}")
    
    # Приводим изображения к одному формату (BGR или BGRA)
    screenshot = manager.screenshot.copy()
    
    # Проверяем и приводим цветовые каналы к одному формату
    if len(screenshot.shape) == 3 and screenshot.shape[2] == 4:
        # Скриншот в формате BGRA (4 канала) -> BGR (3 канала)
        screenshot = screenshot[:, :, :3]
    
    if len(sample_image.shape) == 3 and sample_image.shape[2] == 4:
        # Образец в формате BGRA (4 канала) -> BGR (3 канала)
        sample_image = sample_image[:, :, :3]
    
    # Проверяем и приводим типы данных к одному формату
    if screenshot.dtype != sample_image.dtype:
        # Приводим к общему типу данных
        sample_image = sample_image.astype(np.uint8)
        screenshot = screenshot.astype(np.uint8)
    
    try:
        # Находим все вхождения образца на текущем скриншоте
        result = cv2.matchTemplate(screenshot, sample_image, cv2.TM_CCOEFF_NORMED)
    except cv2.error as e:
        # Более информативное сообщение об ошибке
        raise ValueError(f"Ошибка при сопоставлении шаблонов: {str(e)}. " + 
                        f"Скриншот: {screenshot.shape}, {screenshot.dtype}. " + 
                        f"Образец: {sample_image.shape}, {sample_image.dtype}")
    
    # Получаем порог сходства из настроек
    threshold = settings.OPENCV_MATCH_THRESHOLD
    
    # Находим координаты, где значение результата превышает порог
    locations = np.where(result >= threshold)
    
    # Если образец не найден, возвращаем ошибку
    if len(locations[0]) == 0:
        raise ValueError(f"Образец {sample_id} не найден на текущем экране")
    
    # Если найдено несколько совпадений, выбираем наиболее похожее с помощью векторного сравнения
    if len(locations[0]) > 1:
        best_match_index = 0
        best_similarity = -1
        
        for i in range(len(locations[0])):
            y, x = locations[0][i], locations[1][i]
            
            # Получаем размеры образца
            h, w = sample_image.shape[:2]
            
            # Извлекаем область текущего совпадения
            match_image = screenshot[y:y+h, x:x+w]
            
            # Получаем векторное представление совпадения
            try:
                match_vector = model.get_embedding(match_image)
            except Exception:
                # Если не удалось получить вектор, пропускаем это совпадение
                continue
            
            # Вычисляем косинусное сходство между векторами
            similarity = np.dot(vector, match_vector) / (np.linalg.norm(vector) * np.linalg.norm(match_vector))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_index = i
        
        # Используем координаты лучшего совпадения
        min_y, min_x = locations[0][best_match_index], locations[1][best_match_index]
    else:
        # Если найдено только одно совпадение, используем его координаты
        min_y, min_x = locations[0][0], locations[1][0]
    
    # Получаем размеры образца
    h, w = sample_image.shape[:2]
    
    # Вычисляем центр найденного образца
    center_x = min_x + w // 2
    center_y = min_y + h // 2
    
    return center_x, center_y 