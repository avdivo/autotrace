#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль manager
=============

Модуль содержит класс Manager, который создает ресурсы для процессов 
записи и воспроизведения действий пользователя.
"""

import logging
from typing import Optional, Dict, Any
import numpy as np

import settings
from screen_monitor import ScreenMonitor
from chroma_db import chroma_db as chroma  # Импортируем существующий экземпляр с правильным именем
from exceptions import MonitorError, ChromaDBError, BaseAppError

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
        monitor (ScreenMonitor): Объект класса ScreenMonitor для отслеживания экрана.
    """
    
    def __init__(self) -> None:
        """
        Инициализирует объект Manager.
        
        Создает ресурсы для работы системы записи и воспроизведения.
        """
        self.screen_id: Optional[str] = None
        self.screenshot: Optional[np.ndarray] = None
        self.monitor: ScreenMonitor = ScreenMonitor()
        self._is_running: bool = False
        logger.info("Manager инициализирован")
    
    def start(self) -> None:
        """
        Запускает мониторинг экрана.
        
        Вызывает метод start() объекта monitor, запуская отслеживание экрана.
        
        Raises:
            MonitorError: Если не удалось запустить мониторинг экрана.
        """
        if self._is_running:
            logger.warning("Manager уже запущен")
            return
            
        try:
            logger.info("Запуск мониторинга экрана...")
            self.monitor.start()
            self._is_running = True
            logger.info("Мониторинг экрана успешно запущен")
        except Exception as e:
            logger.error(f"Ошибка при запуске мониторинга экрана: {str(e)}")
            # Останавливаем компоненты, которые могли быть запущены
            self.stop()
            raise MonitorError(f"Не удалось запустить мониторинг экрана: {str(e)}")
    
    def stop(self) -> None:
        """
        Останавливает мониторинг экрана и завершает работу ChromaDB.
        
        Вызывает метод stop() объекта monitor, останавливая отслеживание экрана.
        Вызывает метод shutdown() класса Chroma для остановки сервера БД.
        """
        logger.info("Остановка Manager...")
        
        # Останавливаем мониторинг экрана
        if self.monitor:
            try:
                self.monitor.stop()
                logger.info("Мониторинг экрана остановлен")
            except Exception as e:
                logger.error(f"Ошибка при остановке мониторинга экрана: {str(e)}")
        
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
