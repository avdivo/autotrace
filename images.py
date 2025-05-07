"""
Модуль для работы с изображениями и их векторными представлениями.

Модуль содержит функции для обновления текущего экрана программы,
сохранения области экрана с событием мыши и получения координат
указанного изображения на экране.
"""

import os
import cv2
from typing import Tuple, Optional, Dict

import settings
from chroma_db import chroma_db
from manager import Manager
from exceptions import ScreenCaptureError, ElementNotFoundError
from hash_function import compute_dhash_vector, dhash_vector_to_hex
from ui_detector import UIRegions


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
    
    # Получаем dHash выбранного участка виде вектора
    embedding = [float(bit) for bit in compute_dhash_vector(screen_area)]

    # Проверяем наличие screen_id по вектору
    screen_id = chroma_db.get_screen_id(embedding)
    
    # Если screen_id не найден, создаем новый
    if screen_id is None:
        screen_id = settings.generate_unique_id()
        chroma_db.create_screen(screen_id, embedding)
        
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

    ui_regions = UIRegions(manager.screenshot)  # Создаем объект с регионами
    region = ui_regions.process_click(x, y)  # Поиск региона содержащего элемент, по которому был клик
    hash_region = dhash_vector_to_hex(region.dhash)  # Получаем шестнадцатиричный хэш
    extended_region = ui_regions.merge_nearby_boxes(region, 30)  # Поиск расширенного региона (со стоящими рядом)
    hash_extended_region = dhash_vector_to_hex(extended_region.dhash)  # Получаем шестнадцатиричный хэш

    # Создаем метаданные с текущим screen_id
    metadata = {'screen_id': manager.screen_id,
                'hash_region': hash_region,
                'hash_extended_region': hash_extended_region}

    sample_id = settings.generate_unique_id()
    chroma_db.create_sample(sample_id, metadata)
    
    # # Сохраняем образец в папке sample
    # samples_folder = settings.SAMPLES_DIR
    # os.makedirs(samples_folder, exist_ok=True)
    # sample_path = os.path.join(samples_folder, f"{sample_id}.png")
    # cv2.imwrite(sample_path, sample_image)
    
    return sample_id


def get_xy(manager: Manager, metadata: Dict) -> Tuple[int, int]:
    """
    Находит координаты на экране, где расположено изображение с указанным sample_id.
    
    Args:
        manager: Объект Manager для доступа к компонентам системы
        sample_id: Идентификатор образца
        metadata: Векторное представление образца
        
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

    ui_regions = UIRegions(manager.screenshot)  # Создаем объект с регионами

    # Извлечение хэшей
    hash_region = metadata['hash_region']
    hash_extended_region = metadata['hash_extended_region']

    best_regions = ui_regions.find_best_matching_regions(hash_region)  # Поиск регионов подходящих на образец (по хэшу)

    if len(best_regions) > 1:
        # Если найдено несколько похожих регионов, нужно использовать контекст
        # Второй хэш учитывает рядом стоящие элементы
        region = ui_regions.find_by_extended_regions(best_regions, hash_extended_region)  # Ищем среди регионов лучший
        best_regions = [region] if region else []

    if not best_regions:
        raise ElementNotFoundError("Элемент на экране не найден")

    x, y, w, h = best_regions[0].box

    # Вычисляем центр найденного образца
    center_x = x + w // 2
    center_y = y + h // 2
    
    return center_x, center_y
