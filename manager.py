#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль manager
=============

Модуль содержит класс Manager, который создает ресурсы для процессов 
записи и воспроизведения действий пользователя.
"""
import os
import cv2
import logging
import threading
from typing import Optional, Dict, Any
import numpy as np

import settings
from chroma_db import chroma_db as chroma  # Импортируем существующий экземпляр с правильным именем
from exceptions import MonitorError, ChromaDBError, BaseAppError
from ui_detector import ScreenCapturer
from hash_function import compute_dhash_vector, dhash_vector_to_hex


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Manager:
    """
    Класс для управления ресурсами системы.

    Создает и управляет ресурсами для процессов записи и воспроизведения.
    После создания вызывается start() в try и stop() в finally.

    Attributes:
        screen_id (Optional[str]): Идентификатор текущего экрана.
        screenshot (Optional[np.ndarray]): Изображение экрана в формате numpy.
        chroma: Объект для работы с векторной базой данных ChromaDB.
    """

    def __init__(self, action_name="noname") -> None:
        """
        Инициализирует объект Manager.

        Создает ресурсы для работы системы записи и воспроизведения.
        """
        self.blocked = False  # Блокировка доступа к скриншоту во время его обновления

        self.screen_id: Optional[str] = None
        self.screenshot: Optional[np.ndarray] = None
        self.chroma = chroma  # Добавляем ссылку на глобальный экземпляр chroma_db
        self._is_running: bool = False
        self.timer: Optional[threading.Timer] = None

        self.action_name = action_name  # Название блока команд
        self.step_number = 0  # Номер выполняемого действия
        self.command_name = ""  # Название выполняемого действия (команда)

        logger.info(f"Выполняется {action_name}")

    def _delayed_update(self, delay: float):
        """
        Метод, который будет выполняться по истечении задержки.
        """
        self.screen_id = self.update_screen()

    def screen_update(self, delay: float = 0.4):
        """
        Запускает обновление экрана с задержкой.
        """
        self.blocked = True  #
        if self.timer and self.timer.is_alive():
            self.timer.cancel()

        self.timer = threading.Timer(delay, self._delayed_update, args=(delay,))
        self.timer.start()

    def update_screen(self) -> str:
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
        # Подготовка средства захвата экрана
        capturer = ScreenCapturer(monitor_number=settings.SCREEN_MONITOR_NUMBER)

        # Получаем скриншот
        screenshot = capturer.capture()

        # Сохраняем скриншот в Manager
        self.screenshot = screenshot.copy()

        # Получаем настройки для размера области и её расположения
        area_size = settings.SCREEN_AREA_SIZE
        area_x, area_y = settings.TOP_LEFT_CORNER

        # Извлекаем область верхнего левого угла (или другую, заданную в настройках)
        screen_area = screenshot[area_y:area_y + area_size[1], area_x:area_x + area_size[0]]

        # Получаем dHash выбранного участка виде вектора
        embedding = [float(bit) for bit in compute_dhash_vector(screen_area)]

        # Проверяем наличие screen_id по вектору
        screen_id = chroma.get_screen_id(embedding)

        # Если screen_id не найден, создаем новый
        if screen_id is None:
            screen_id = settings.generate_unique_id()
            chroma.create_screen(screen_id, embedding)

        # # Сохраняем полный скриншот в папке screenshots
        # screenshots_folder = settings.SCREENSHOTS_DIR
        # os.makedirs(screenshots_folder, exist_ok=True)
        # screenshot_path = os.path.join(screenshots_folder, f"{screen_id}.png")
        # cv2.imwrite(screenshot_path, screenshot)

        # Сохраняем screen_id в Manager и возвращаем его
        self.screen_id = screen_id
        return screen_id

    def stop(self) -> None:
        """
        Останавливает мониторинг экрана и завершает работу ChromaDB.
        
        Вызывает метод stop() объекта monitor, останавливая отслеживание экрана.
        Вызывает метод shutdown() класса Chroma для остановки сервера БД.
        """
        logger.info("Остановка Manager...")

        """Останавливает таймер и завершает работу менеджера."""
        if self.timer and self.timer.is_alive():
            self.timer.cancel()

        # Останавливаем сервер ChromaDB
        try:
            chroma.shutdown()
            logger.info("Сервер ChromaDB остановлен")
        except ChromaDBError as e:
            logger.error(f"Ошибка при остановке сервера ChromaDB: {str(e)}")
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при остановке сервера ChromaDB: {str(e)}")
        
        # Сбрасываем состояние
        self.screenshot = None
        self.screen_id = None
        self._is_running = False
        
        logger.info("Manager успешно остановлен")

    def add_command(self, command: str = '') -> int:
        """
        Добавляет команду, которая обрабатывается.
        Если команда не пустая увеличивается счетчик команд.

        Args:
            Команда.

        Returns:
            Состояние счетчика команд.
        """
        if command:
            self.command_name = command
            self.step_number += 1

        return self.step_number

    def report(self, img: np.ndarray = None,
               is_report: bool = settings.OUT_REPORT,
               is_save_scr: bool = settings.SAVE_SCREENSHOT,
               folder_to_save: str = settings.SCREENSHOTS_DIR,
               green_blocks: list = None,
               red_block: tuple = None,
               blue_block: tuple = None
               ):
        """
        Выводит в лог команду которая выполняется.
        Сохраняет переданное изображение или подготовленное
        в папку скриншотов.
        Логика подготовки изображения:
        1. Если передано, то используется оно
        2. Если не передано, то используется хранимый скриншот
        3. Если переданы блоки (набор зеленых, красный, синий) они наносятся на изображение
        4. Если ничего из перечисленного не передано, то изображение не сохраняется.

        Работа функции задается настройками, но может гибко меняться аргументами.

        Args:
            img - изображение которое нужно сохранить
            is_report - Нужно ли выводить команду в лог
            is_save_scr - Нужно ли сохранять скриншот
            folder_to_save - папка для сохранения изображения.
            green_blocks - список кортежей с координатами и размерами (x, y, w, h) зеленых прямоугольников
            red_block - кортеж с координатами и размерами (x, y, w, h) красного прямоугольника
            blue_block - - кортеж с координатами и размерами (x, y, w, h) синего прямоугольника
        """
        # Сохраняем изображение в заданной папке
        if is_save_scr:
            # В любом случае что-то нужно сохранить
            if img is None:
                # Изображение не передано, берем скриншот
                img = self.screenshot.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Отмечаем на изображении регионы
            # Зеленые
            if green_blocks is not None:
                for region in green_blocks:
                    x, y, w, h = region.box  # Извлекаем кортеж (x, y, w, h)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Красный
            if red_block is not None:
                x, y, w, h = red_block
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Синий
            if red_block is not None:
                x, y, w, h = blue_block
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Сохраняем изображение
            os.makedirs(folder_to_save, exist_ok=True)
            screenshot_path = os.path.join(folder_to_save, f"{self.step_number}_{self.action_name}.png")
            cv2.imwrite(screenshot_path, img)

        # Вывод команды
        if is_report:
            print(f"{self.step_number}. {self.command_name}")

    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает текущее состояние Manager.
        
        Returns:
            Dict[str, Any]: Словарь с информацией о текущем состоянии:
                - is_running (bool): Активен ли Manager
                - screen_id (Optional[str]): Текущий идентификатор экрана
                - has_screenshot (bool): Есть ли текущий скриншот
        """
        return {
            "is_running": self._is_running,
            "screen_id": self.screen_id,
            "has_screenshot": self.screenshot is not None
        }
    
    def is_running(self) -> bool:
        """
        Проверяет, запущен ли Manager.
        
        Returns:
            bool: True, если Manager активен, иначе False.
        """
        return self._is_running
