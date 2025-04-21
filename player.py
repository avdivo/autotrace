#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для воспроизведения действий пользователя
================================================

Модуль предоставляет класс Player для воспроизведения записанных действий пользователя 
по одиночным командам или их списку.
"""

import time
import sys
import re
from typing import List, Optional, Dict, Any, Union
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController

# Импорт из других модулей проекта
import settings
from exceptions import (
    CommandNotFoundError,
    ScreenMismatchError, 
    InvalidCommandError,
    PlaybackError
)

# Функции-заглушки для режима тестирования
def mock_screen_update(manager: Any) -> str:
    """Заглушка для функции screen_update при автономном запуске."""
    return 'test_screen_id'

def mock_get_xy(manager: Any, sample_id: str, vector: List[float]) -> tuple[int, int]:
    """Заглушка для функции get_xy при автономном запуске."""
    return (100, 100)


class Player:
    """
    Класс для воспроизведения действий пользователя.
    
    Воспроизводит действия пользователя по одной команде или списком команд.
    Поддерживает действия с клавиатурой, мышью и специальные команды.
    """
    
    def __init__(self, manager: Any):
        """
        Инициализация класса Player.
        
        Параметры:
            manager (Any): Объект менеджера, управляющий ресурсами для 
                          процессов записи и воспроизведения.
        """
        self.manager = manager
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        
        # Словарь для маппинга специальных клавиш по их строковому представлению
        self.key_mapping = {
            'alt': Key.alt,
            'alt_l': Key.alt_l,
            'alt_r': Key.alt_r,
            'ctrl': Key.ctrl,
            'ctrl_l': Key.ctrl_l,
            'ctrl_r': Key.ctrl_r,
            'shift': Key.shift,
            'shift_l': Key.shift_l,
            'shift_r': Key.shift_r,
            'backspace': Key.backspace,
            'caps_lock': Key.caps_lock,
            'delete': Key.delete,
            'down': Key.down,
            'end': Key.end,
            'enter': Key.enter,
            'esc': Key.esc,
            'escape': Key.esc,
            'f1': Key.f1,
            'f2': Key.f2,
            'f3': Key.f3,
            'f4': Key.f4,
            'f5': Key.f5,
            'f6': Key.f6,
            'f7': Key.f7,
            'f8': Key.f8,
            'f9': Key.f9,
            'f10': Key.f10,
            'f11': Key.f11,
            'f12': Key.f12,
            'home': Key.home,
            'insert': Key.insert,
            'left': Key.left,
            'menu': Key.menu,
            'num_lock': Key.num_lock,
            'page_down': Key.page_down,
            'page_up': Key.page_up,
            'pause': Key.pause,
            'print_screen': Key.print_screen,
            'right': Key.right,
            'scroll_lock': Key.scroll_lock,
            'space': Key.space,
            'tab': Key.tab,
            'up': Key.up
        }
        
        # Словарь для маппинга кнопок мыши
        self.mouse_button_mapping = {
            'left': Button.left,
            'right': Button.right,
            'middle': Button.middle
        }
    
    def _get_key(self, key_name: str) -> Union[Key, str]:
        """
        Преобразует строковое представление клавиши в объект Key.
        
        Параметры:
            key_name (str): Строковое представление клавиши.
            
        Возвращает:
            Union[Key, str]: Объект Key или строку, если клавиша не найдена в маппинге.
        """
        return self.key_mapping.get(key_name.lower(), key_name)
    
    def play_one(self, command: str) -> None:
        """
        Воспроизводит одну команду.
        
        Команды разделяются на блоки:
        - kbd_click_(keys)_event_id — нажать и отпустить клавишу
        - kbd_combo_(keys)_event_id — сочетание клавиш
        - mouse_click_left/right_event_id — клик левой/правой клавишей мыши
        - mouse_dblclick_left_event_id — двойной клик левой клавишей мыши
        - mouse_scroll_x,y_event_id — прокрутка колесика мыши
        
        Параметры:
            command (str): Строка команды для выполнения.
            
        Вызывает:
            InvalidCommandError: Если формат команды некорректен.
            CommandNotFoundError: Если команда не найдена в базе данных.
            ScreenMismatchError: Если текущий экран не соответствует требуемому.
        """
        # Определяем нужные функции в зависимости от режима работы
        is_standalone = hasattr(self.manager, 'is_standalone') and self.manager.is_standalone
        
        # Импортируем функции или используем заглушки
        if is_standalone:
            screen_update = mock_screen_update
            get_xy = mock_get_xy
        else:
            # Импортируем функции только если не в автономном режиме
            from images import screen_update, get_xy
        
        # Обновляем текущий экран перед выполнением команды
        current_screen_id = screen_update(self.manager)
        
        # Используем регулярные выражения для извлечения блока и действия команды
        match = re.match(r'^(\w+)_(\w+)_(.+)$', command)
        if not match:
            raise InvalidCommandError(f"Некорректный формат команды: {command}")
        
        command_block = match.group(1)
        command_action = match.group(2)
        command_rest = match.group(3)  # Оставшаяся часть команды
        
        # Получаем идентификатор (последняя часть команды после последнего подчеркивания)
        sample_id = command.split('_')[-1]
        
        # Специальная обработка для автономного запуска
        if is_standalone:
            if command_block == "kbd":
                self._execute_kbd_command(command_action, command)
                return
            elif command_block == "mouse":
                # Создаем заглушку для sample_data
                sample_data = {'embeddings': [[0.1, 0.2, 0.3]], 'metadatas': [{'screen_id': current_screen_id}]}
                self._execute_mouse_command(command_action, command, 'test_sample_id', sample_data)
                return
            else:
                raise InvalidCommandError(f"Неизвестный блок команды: {command_block}")
        
        # Получаем документ из базы данных
        sample_data = self.manager.chroma.get_sample(sample_id)
        
        if sample_data is None:
            raise CommandNotFoundError(f"Команда не найдена в базе данных: {command}")
        
        # Проверяем соответствие текущего экрана
        metadata = sample_data.get('metadatas', [{}])[0]
        screen_id_from_db = metadata.get('screen_id')
        
        if screen_id_from_db != current_screen_id:
            raise ScreenMismatchError(
                f"Текущий экран ({current_screen_id}) не соответствует требуемому ({screen_id_from_db})"
            )
        
        # Выполняем команды в зависимости от блока
        if command_block == "kbd":
            self._execute_kbd_command(command_action, command)
        elif command_block == "mouse":
            self._execute_mouse_command(command_action, command, sample_id, sample_data)
        else:
            raise InvalidCommandError(f"Неизвестный блок команды: {command_block}")
    
    def _execute_kbd_command(self, action: str, command: str) -> None:
        """
        Выполняет команды клавиатуры.
        
        Параметры:
            action (str): Действие (click или combo).
            command (str): Полная строка команды.
        """
        if action == "click":
            # Используем регулярное выражение для извлечения ключа из скобок
            match = re.search(r'kbd_click_\(([^)]+)\)_', command)
            if match:
                key_name = match.group(1)
                key = self._get_key(key_name)
                
                # Нажимаем и отпускаем клавишу
                self.keyboard.press(key)
                time.sleep(settings.PLAYER_KEY_PRESS_DELAY)  # Используем задержку из настроек
                self.keyboard.release(key)
            else:
                raise InvalidCommandError(f"Некорректный формат команды kbd_click: {command}")
                
        elif action == "combo":
            # Используем регулярное выражение для извлечения ключей из скобок
            match = re.search(r'kbd_combo_\(([^)]+)\)_', command)
            if match:
                key_names = match.group(1).split()  # Разделяем по пробелам
                keys = [self._get_key(k) for k in key_names]
                
                # Нажимаем все клавиши в комбинации
                for key in keys:
                    self.keyboard.press(key)
                
                time.sleep(settings.PLAYER_KEY_PRESS_DELAY)  # Используем задержку из настроек
                
                # Отпускаем все клавиши в обратном порядке
                for key in reversed(keys):
                    self.keyboard.release(key)
            else:
                raise InvalidCommandError(f"Некорректный формат команды kbd_combo: {command}")
    
    def _execute_mouse_command(self, action: str, command: str, sample_id: str, 
                              sample_data: Dict[str, Any]) -> None:
        """
        Выполняет команды мыши.
        
        Параметры:
            action (str): Действие мыши (click, dblclick, scroll).
            command (str): Полная строка команды.
            sample_id (str): Идентификатор образца.
            sample_data (Dict[str, Any]): Данные образца.
        """
        # Определяем функцию в зависимости от режима работы
        is_standalone = hasattr(self.manager, 'is_standalone') and self.manager.is_standalone
        
        if is_standalone:
            get_xy = mock_get_xy
        else:
            from images import get_xy
        
        if action == "click":
            # Извлекаем кнопку мыши из команды
            match = re.search(r'mouse_click_(\w+)_', command)
            if not match:
                raise InvalidCommandError(f"Некорректный формат команды mouse_click: {command}")
                
            button_str = match.group(1)
            button = self.mouse_button_mapping.get(button_str)
            
            if not button:
                raise InvalidCommandError(f"Некорректная кнопка мыши: {button_str}")
            
            # Определяем координаты для клика
            vector = sample_data.get('embeddings', [[]])[0]
            x, y = get_xy(self.manager, sample_id, vector)
            
            # Перемещаем мышь и выполняем клик
            self.mouse.position = (x, y)
            time.sleep(settings.PLAYER_MOUSE_MOVE_DELAY)  # Используем задержку из настроек
            self.mouse.click(button)
            
        elif action == "dblclick":
            # Извлекаем кнопку мыши из команды
            match = re.search(r'mouse_dblclick_(\w+)_', command)
            if not match:
                raise InvalidCommandError(f"Некорректный формат команды mouse_dblclick: {command}")
                
            button_str = match.group(1)
            button = self.mouse_button_mapping.get(button_str)
            
            if not button:
                raise InvalidCommandError(f"Некорректная кнопка мыши: {button_str}")
            
            # Определяем координаты для клика
            vector = sample_data.get('embeddings', [[]])[0]
            x, y = get_xy(self.manager, sample_id, vector)
            
            # Перемещаем мышь и выполняем двойной клик
            self.mouse.position = (x, y)
            time.sleep(settings.PLAYER_MOUSE_MOVE_DELAY)  # Используем задержку из настроек
            self.mouse.click(button, 2)
            
        elif action == "scroll":
            # Извлекаем значения прокрутки из команды
            match = re.search(r'mouse_scroll_([^_]+)_', command)
            if not match:
                raise InvalidCommandError(f"Некорректный формат команды mouse_scroll: {command}")
                
            scroll_values_str = match.group(1)
            scroll_values = scroll_values_str.split(',')
            
            if len(scroll_values) != 2:
                raise InvalidCommandError(f"Некорректные значения прокрутки: {scroll_values_str}")
            
            try:
                dx, dy = float(scroll_values[0]), float(scroll_values[1])
            except ValueError:
                raise InvalidCommandError(f"Некорректные значения прокрутки: {scroll_values_str}")
            
            # Определяем координаты для прокрутки
            vector = sample_data.get('embeddings', [[]])[0]
            x, y = get_xy(self.manager, sample_id, vector)
            
            # Перемещаем мышь и выполняем прокрутку
            self.mouse.position = (x, y)
            time.sleep(settings.PLAYER_MOUSE_MOVE_DELAY)  # Используем задержку из настроек
            self.mouse.scroll(dx, dy)
       
    def play_all(self, commands: List[str]) -> None:
        """
        Выполняет список команд последовательно.
        
        При возникновении ошибки выводит сообщение об ошибке и продолжает выполнение.
        
        Параметры:
            commands (List[str]): Список команд для выполнения.
        """
        for i, command in enumerate(commands):
            try:
                self.play_one(command)
                # Небольшая задержка между командами для стабильности
                time.sleep(settings.PLAYER_COMMAND_DELAY)  # Используем задержку из настроек
            except Exception as e:
                print(f"Ошибка при выполнении команды {i+1} ({command}): {str(e)}")


# Заглушка для менеджера при автономном запуске
class MockManager:
    """Заглушка для менеджера при автономном запуске модуля."""
    
    def __init__(self):
        """Инициализация заглушки менеджера."""
        self.is_standalone = True
        self.screen_id = 'test_screen_id'
        self.screenshot = None
        
        # Заглушка для chroma
        class MockChroma:
            def get_sample(self, sample_id):
                return {
                    'ids': [sample_id],
                    'embeddings': [[0.1, 0.2, 0.3]],
                    'metadatas': [{'screen_id': 'test_screen_id'}]
                }
        
        self.chroma = MockChroma()


# Код для автономного запуска модуля
if __name__ == "__main__":
    try:
        print("Запуск автономного тестирования модуля player.py")
        
        # Создаём заглушку менеджера
        mock_manager = MockManager()
        
        # Создаём объект Player
        player = Player(mock_manager)
        
        # Создаем список команд для набора слова "Привет"
        print("Подготовка к набору слова 'Привет'...")
        
        # Даём пользователю время переключиться на нужное окно
        print(f"У вас есть {settings.PLAYER_TEST_WAIT_TIME} секунд, чтобы переключиться на нужное окно...")
        for i in range(settings.PLAYER_TEST_WAIT_TIME, 0, -1):  # Используем время ожидания из настроек
            print(f"{i}...")
            time.sleep(1)
        
        # Генерируем уникальные идентификаторы для команд
        command_ids = [f"test_{i}" for i in range(1, 8)]
        
        # Формируем команды для ввода слова "Привет" с использованием правильного формата
        commands = [
            # Для заглавной буквы П - комбинация Shift+п
            f"kbd_combo_(shift п)_{command_ids[0]}",
            f"kbd_click_(р)_{command_ids[1]}",
            f"kbd_click_(и)_{command_ids[2]}",
            f"kbd_click_(в)_{command_ids[3]}",
            f"kbd_click_(е)_{command_ids[4]}",
            f"kbd_click_(т)_{command_ids[5]}",
        ]
        
        print(f"Выполнение команд для набора слова 'Привет'...")
        player.play_all(commands)
        
        print("Тестирование завершено!")
        
    except Exception as e:
        print(f"Ошибка при тестировании: {str(e)}")
        