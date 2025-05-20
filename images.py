"""
Модуль для работы с изображениями и их векторными представлениями.

Модуль содержит функции для обновления текущего экрана программы,
сохранения области экрана с событием мыши и получения координат
указанного изображения на экране.
"""
import time

import cv2
from typing import Tuple, Optional, Dict

import settings
from chroma_db import chroma_db
from manager import Manager
from exceptions import ScreenCaptureError, ElementNotFoundError
from hash_function import compute_dhash_vector, dhash_vector_to_hex
from ui_detector import UIRegions


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

    # print("Хэш основного региона", hash_region, "\n", region.dhash)

    # Создаем метаданные с текущим screen_id
    metadata = {'screen_id': manager.screen_id,
                'hash_region': hash_region,
                'hash_extended_region': hash_extended_region}

    sample_id = settings.generate_unique_id()
    chroma_db.create_sample(sample_id, metadata)

    # ДЛЯ ОТЛАДКИ. Выводим команду, сохраняем скриншот
    manager.report(green_blocks=ui_regions.regions, red_block=region.box, blue_block=extended_region.box)

    # ------------------------------------------------------------------------------
    # Вывод найденного элемента по клику (красным - основной, зеленый - расширенный)
    # screen_image = manager.screenshot.copy()
    # x, y, w, h = region.box
    # cv2.rectangle(screen_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # x, y, w, h = extended_region.box
    # cv2.rectangle(screen_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.namedWindow("Element", cv2.WINDOW_FREERATIO)  # Окно можно изменять
    # screen_image = cv2.cvtColor(screen_image, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Element", screen_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return sample_id


def get_xy(manager: Manager, metadata: Dict) -> Tuple[int, int]:
    """
    Находит координаты на  экране, где расположено изображение с указанным sample_id.
    
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
    # Извлечение хэшей
    hash_region = metadata['hash_region']
    hash_extended_region = metadata['hash_extended_region']

    # Проверяем наличие элемента на экране
    if manager.screenshot is None:
        raise ScreenCaptureError("Текущий скриншот недоступен")

    # Проверяем наличие элемента на экране
    start_wait = time.time()
    best_regions = []
    while len(best_regions) == 0 and time.time() - start_wait < settings.PLAYER_ELEMENT_WAIT_TIME:
        ui_regions = UIRegions(manager.screenshot)  # Создаем объект с регионами
        best_regions = ui_regions.find_best_matching_regions(hash_region)  # Поиск регионов подходящих на образец (по хэшу)
        manager.screen_update(0)
        time.sleep(0.3)

    # ДЛЯ ОТЛАДКИ. Выводим команду, сохраняем скриншот
    manager.report(green_blocks=ui_regions.regions)

    if not best_regions:
        raise ElementNotFoundError("Элемент на экране не найден")

    if len(best_regions) > 1:
        # Если найдено несколько похожих регионов, нужно использовать контекст
        # Второй хэш учитывает рядом стоящие элементы
        region = ui_regions.find_by_extended_regions(best_regions, hash_extended_region)  # Ищем среди регионов лучший
        best_regions = [region] if region else []

    x, y, w, h = best_regions[0].box

    # Вычисляем центр найденного образца
    center_x = x + w // 2
    center_y = y + h // 2
    
    return center_x, center_y

