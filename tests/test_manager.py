#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Тесты для модуля manager
=======================

Простые тесты для проверки функциональности класса Manager.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import numpy as np
import sys
import os

# Добавляем корневую директорию проекта в путь для импорта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируем реальные модули
from exceptions import MonitorError, ChromaDBError, BaseAppError
from screen_monitor import ScreenMonitor
from chroma_db import chroma_db as chroma
from manager import Manager

class TestManager(unittest.TestCase):
    """Тесты для класса Manager."""

    def setUp(self):
        """Подготовка тестового окружения перед каждым тестом."""
        # Создаем патчи для методов, которые не должны выполняться в тестах
        self.screen_monitor_start_patcher = patch.object(ScreenMonitor, 'start')
        self.screen_monitor_stop_patcher = patch.object(ScreenMonitor, 'stop')
        self.chroma_shutdown_patcher = patch.object(chroma, 'shutdown')
        
        # Активируем патчи
        self.mock_screen_monitor_start = self.screen_monitor_start_patcher.start()
        self.mock_screen_monitor_stop = self.screen_monitor_stop_patcher.start()
        self.mock_chroma_shutdown = self.chroma_shutdown_patcher.start()
        
        # Создаем экземпляр Manager для тестов
        self.manager = Manager()
    
    def tearDown(self):
        """Очистка тестового окружения после каждого теста."""
        # Останавливаем патчи
        self.screen_monitor_start_patcher.stop()
        self.screen_monitor_stop_patcher.stop()
        self.chroma_shutdown_patcher.stop()
    
    def test_init(self):
        """Тестирование инициализации Manager."""
        # Проверяем, что атрибуты инициализированы корректно
        self.assertIsNone(self.manager.screen_id)
        self.assertIsNone(self.manager.screenshot)
        self.assertIsInstance(self.manager.monitor, ScreenMonitor)
        self.assertFalse(self.manager._is_running)
    
    def test_start(self):
        """Тестирование метода start."""
        # Вызываем метод start
        self.manager.start()
        
        # Проверяем, что метод start вызван у монитора
        self.mock_screen_monitor_start.assert_called_once()
        
        # Проверяем, что флаг запуска установлен
        self.assertTrue(self.manager._is_running)
    
    def test_start_already_running(self):
        """Тестирование метода start при уже запущенном менеджере."""
        # Устанавливаем флаг запуска
        self.manager._is_running = True
        
        # Вызываем метод start
        self.manager.start()
        
        # Проверяем, что метод start не вызывался у монитора
        self.mock_screen_monitor_start.assert_not_called()
    
    def test_start_exception(self):
        """Тестирование обработки исключений в методе start."""
        # Настраиваем монитор, чтобы он вызывал исключение
        self.mock_screen_monitor_start.side_effect = Exception("Тестовая ошибка")
        
        # Проверяем, что метод start вызывает MonitorError
        with self.assertRaises(MonitorError):
            self.manager.start()
        
        # Проверяем, что флаг запуска не установлен
        self.assertFalse(self.manager._is_running)
        
        # Проверяем, что метод stop был вызван для очистки ресурсов
        self.mock_screen_monitor_stop.assert_called_once()
    
    def test_stop(self):
        """Тестирование метода stop."""
        # Устанавливаем начальное состояние
        self.manager._is_running = True
        self.manager.screen_id = "test_screen_id"
        self.manager.screenshot = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Вызываем метод stop
        self.manager.stop()
        
        # Проверяем, что метод stop вызван у монитора
        self.mock_screen_monitor_stop.assert_called_once()
        
        # Проверяем, что метод shutdown вызван у ChromaDB
        self.mock_chroma_shutdown.assert_called_once()
        
        # Проверяем, что состояние сброшено
        self.assertFalse(self.manager._is_running)
        self.assertIsNone(self.manager.screen_id)
        self.assertIsNone(self.manager.screenshot)
    
    def test_stop_exception_monitor(self):
        """Тестирование обработки исключений монитора в методе stop."""
        # Настраиваем монитор, чтобы он вызывал исключение
        self.mock_screen_monitor_stop.side_effect = Exception("Тестовая ошибка монитора")
        
        # Устанавливаем начальное состояние
        self.manager._is_running = True
        
        # Вызываем метод stop
        self.manager.stop()
        
        # Проверяем, что несмотря на ошибку, метод shutdown был вызван у ChromaDB
        self.mock_chroma_shutdown.assert_called_once()
        
        # Проверяем, что состояние было сброшено
        self.assertFalse(self.manager._is_running)
    
    def test_stop_exception_chroma(self):
        """Тестирование обработки исключений ChromaDB в методе stop."""
        # Настраиваем ChromaDB, чтобы он вызывал исключение
        self.mock_chroma_shutdown.side_effect = Exception("Тестовая ошибка ChromaDB")
        
        # Устанавливаем начальное состояние
        self.manager._is_running = True
        
        # Вызываем метод stop
        self.manager.stop()
        
        # Проверяем, что метод stop был вызван у монитора
        self.mock_screen_monitor_stop.assert_called_once()
        
        # Проверяем, что состояние было сброшено несмотря на ошибку
        self.assertFalse(self.manager._is_running)
    
    def test_get_status(self):
        """Тестирование метода get_status."""
        # Устанавливаем начальное состояние
        self.manager._is_running = True
        self.manager.screen_id = "test_screen_id"
        self.manager.screenshot = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Получаем статус
        status = self.manager.get_status()
        
        # Проверяем результат
        expected_status = {
            "is_running": True,
            "screen_id": "test_screen_id",
            "has_screenshot": True
        }
        self.assertEqual(status, expected_status)
    
    def test_is_running(self):
        """Тестирование метода is_running."""
        # Проверяем изначальное состояние
        self.assertFalse(self.manager.is_running())
        
        # Меняем состояние
        self.manager._is_running = True
        
        # Проверяем измененное состояние
        self.assertTrue(self.manager.is_running())

if __name__ == '__main__':
    unittest.main() 