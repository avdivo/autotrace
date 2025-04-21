#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Тесты для модуля screen_monitor
=============================

Модуль содержит тесты для проверки функциональности класса ScreenMonitor из модуля screen_monitor.
Тесты проверяют корректность инициализации, запуска/остановки монитора экрана и работы с shared memory.
"""

import pytest
import time
import numpy as np
import multiprocessing
from multiprocessing import shared_memory
import os
from screen_monitor import ScreenMonitor
from exceptions import ScreenCaptureError, MemoryError, MonitorError
import settings

class TestScreenMonitor:
    """Тесты для класса ScreenMonitor."""
    
    @pytest.fixture(scope="function")
    def screen_monitor(self):
        """Фикстура для создания экземпляра класса ScreenMonitor."""
        # Используем уникальное имя для shared memory в тестах, чтобы избежать конфликтов
        monitor = ScreenMonitor(
            interval=0.5,  # Более частые скриншоты для тестирования
            monitor_number=settings.SCREEN_MONITOR_NUMBER,
            shared_memory_name=f"test_screen_capture_{os.getpid()}",
            width=1920,  # Full HD ширина
            height=1080   # Full HD высота
        )
        yield monitor
        
        # Очистка после тестов
        monitor.stop()
    
    def test_initialization(self, screen_monitor):
        """Проверка корректности инициализации ScreenMonitor."""
        assert screen_monitor is not None
        assert screen_monitor.interval == 0.5
        assert screen_monitor.monitor_number == settings.SCREEN_MONITOR_NUMBER
        assert screen_monitor.width == 1920
        assert screen_monitor.height == 1080
        assert screen_monitor.running == False
        assert screen_monitor.process is None
        assert screen_monitor.shm is not None
        assert screen_monitor.buffer_shape == (1080, 1920, 4)
        assert screen_monitor.buffer_size == 1080 * 1920 * 4
    
    def test_start_stop(self, screen_monitor):
        """Проверка запуска и остановки монитора экрана."""
        # Тест запуска
        screen_monitor.start()
        assert screen_monitor.running == True
        assert screen_monitor.process is not None
        assert screen_monitor.process.is_alive() == True
        
        # Даем процессу время для захвата хотя бы одного скриншота
        time.sleep(1)
        
        # Тест остановки
        screen_monitor.stop()
        assert screen_monitor.running == False
        assert screen_monitor.process is None
        assert screen_monitor.shm is None
    
    def test_get_screenshot(self, screen_monitor):
        """Проверка получения скриншота из shared memory."""
        screen_monitor.start()
        
        # Даем время для захвата скриншота
        time.sleep(1)
        
        # Получаем скриншот
        screenshot = screen_monitor.get_screenshot()
        
        # Останавливаем монитор
        screen_monitor.stop()
        
        # Проверяем, что скриншот получен и имеет правильный формат
        assert screenshot is not None
        assert isinstance(screenshot, np.ndarray)
        assert screenshot.shape == (1080, 1920, 4)  # Формат BGRA для Full HD
        
        # Проверяем, что скриншот не пустой (содержит хоть какие-то ненулевые значения)
        assert not np.all(screenshot == 0), "Скриншот не должен быть пустым (все пиксели нулевые)"
    
    def test_clear_shared_memory(self, screen_monitor):
        """Проверка очистки shared memory."""
        # Запускаем монитор и получаем скриншот
        screen_monitor.start()
        time.sleep(1)
        
        # Получаем скриншот
        screenshot = screen_monitor.get_screenshot()
        assert screenshot is not None
        assert isinstance(screenshot, np.ndarray)
        assert screenshot.shape == (1080, 1920, 4)  # Формат BGRA для Full HD
        
        # Проверяем, что скриншот не пустой
        assert not np.all(screenshot == 0), "Скриншот не должен быть пустым (все пиксели нулевые)"
        
        # После получения скриншота shared memory должна быть уже очищена,
        # так как метод get_screenshot вызывает clear_shared_memory
        
        # Проверяем, что следующий вызов get_screenshot вернет None, так как память очищена
        screenshot = screen_monitor.get_screenshot()
        assert screenshot is None
        
        screen_monitor.stop()
    
    def test_error_handling(self):
        """Проверка обработки ошибок."""
        # Проверяем ошибку при указании несуществующего монитора
        # Ошибка должна возникать при инициализации
        with pytest.raises(ScreenCaptureError):
            # Предполагается, что монитор с номером 999 не существует
            monitor = ScreenMonitor(
                monitor_number=999,
                shared_memory_name=f"test_error_capture_{os.getpid()}",
                width=1920,
                height=1080
            )

    def test_multiple_monitor_instances(self):
        """Проверка работы с несколькими экземплярами ScreenMonitor."""
        # Создаем два монитора с разными именами shared memory
        monitor1 = ScreenMonitor(
            interval=0.5,
            shared_memory_name=f"test_multi_monitor1_{os.getpid()}",
            width=1920,
            height=1080
        )
        
        monitor2 = ScreenMonitor(
            interval=0.5,
            shared_memory_name=f"test_multi_monitor2_{os.getpid()}",
            width=1920,
            height=1080
        )
        
        try:
            # Запускаем оба монитора
            monitor1.start()
            monitor2.start()
            
            # Даем время для захвата скриншотов
            time.sleep(1)
            
            # Получаем скриншоты с обоих мониторов
            screenshot1 = monitor1.get_screenshot()
            screenshot2 = monitor2.get_screenshot()
            
            # Проверяем, что скриншоты получены
            assert screenshot1 is not None
            assert screenshot2 is not None
            
            # Проверяем, что скриншоты не пустые
            assert not np.all(screenshot1 == 0), "Скриншот монитора 1 не должен быть пустым"
            assert not np.all(screenshot2 == 0), "Скриншот монитора 2 не должен быть пустым"
            
            # Проверяем, что это разные объекты
            assert id(screenshot1) != id(screenshot2)
            
            # Проверяем, что они имеют правильный формат
            assert screenshot1.shape == (1080, 1920, 4)
            assert screenshot2.shape == (1080, 1920, 4)
            
        finally:
            # Останавливаем мониторы и освобождаем ресурсы
            monitor1.stop()
            monitor2.stop()

if __name__ == "__main__":
    pytest.main(["-v"]) 