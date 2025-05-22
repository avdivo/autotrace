#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для воспроизведения действий пользователя
================================================

Модуль предоставляет класс Player для воспроизведения записанных действий пользователя 
по одиночным командам или их списку.
"""

import re
import sys
import time
from typing import List, Optional, Dict, Any, Union
from pynput import keyboard
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController
import pyautogui  # Добавляем импорт PyAutoGUI

# Импорт из других модулей проекта
import settings
from exceptions import (
    CommandNotFoundError,
    ScreenMismatchError, 
    InvalidCommandError,
    PlaybackError
)
from chroma_db import chroma_db  # Добавляем прямой импорт chroma_db


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

        # Запуск слушателя клавиш в фоне
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

    def on_press(self, key):
        """Слушатель клавиатуры для остановки выполнения программы."""
        if key == keyboard.Key.esc:
            current_time = time.time()
            if current_time - self.manager.last_esc_time <= 0.5:
                self.manager.stop_flag = True
                return False  # Остановить слушатель
            self.manager.last_esc_time = current_time

    def _get_key(self, key_name: str) -> Union[Key, str]:
        """
        Преобразует строковое представление клавиши в объект Key.
        
        Параметры:
            key_name (str): Строковое представление клавиши.
            
        Возвращает:
            Union[Key, str]: Объект Key или строку, если клавиша не найдена в маппинге.
        """
        return self.key_mapping.get(key_name.lower(), key_name)

    def wait_screen(self, screen_id: str):
        """
        Желает скриншоты экрана и проверяет его на соответствие переданному идентификатору
        экрана. По истечении лимита времени возвращает ошибку.

        Параметры:
            Идентификатор экрана.
        """
        # Ожидаем соответствие id текущему экрану
        start_wait = time.time()
        while screen_id != self.manager.screen_id and time.time() - start_wait < settings.PLAYER_SCREEN_WAIT_TIME:
            self.manager.screen_update(0)  # Экран должен обновиться после события
            # time.sleep(0.01)

        if screen_id != self.manager.screen_id:
            raise ScreenMismatchError(
                f"Текущий экран ({self.manager.screen_id}) не соответствует требуемому ({screen_id})"
            )

        # Если экран получен, обновляем еще раз, с задержкой, чтобы дать возможность ему прогрузиться
        self.manager.screen_update(settings.PLAYER_SCREEN_WAIT_LOAD)

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
            from images import get_xy

        self.manager.add_command(command)  # Обновляем данные о событии в менеджере

        # Определяем тип команды и действие
        command_parts = command.split('_')
        
        # Минимум должно быть 3 части: блок_действие_данные
        if len(command_parts) < 3:
            raise InvalidCommandError(f"Некорректный формат команды: {command}")
            
        command_block = command_parts[0]  # kbd или mouse
        command_action = command_parts[1]  # click, combo, и т.д.
        
        # Обработка команд клавиатуры
        if command_block == "kbd":
            # Получаем идентификатор из последней части команды
            screen_id = command_parts[-1]

            self.wait_screen(screen_id)  # Ожидаем переход к нужному экрану
            self._execute_kbd_command(command_action, command)
            self.manager.report()  # ДЛЯ ОТЛАДКИ. Выводим команду, сохраняем скриншот

            return
        
        # Обработка команд мыши    
        elif command_block == "mouse":
            # Для команд мыши формат: mouse_action_button_id
            # Получаем идентификатор из последней части команды
            sample_id = command_parts[-1]
            
            # Специальная обработка для автономного запуска
            if is_standalone:
                # Создаем заглушку для sample_data
                sample_data = {'embedding': [0.1, 0.2, 0.3], 'metadata': {'screen_id': self.manager.screen_id}}
                self._execute_mouse_command(command_action, command, sample_id, sample_data)
                return
            
            # Получаем данные образца из базы
            sample_data = chroma_db.get_sample(sample_id)
            
            if sample_data is None:
                raise CommandNotFoundError(f"Команда не найдена в базе данных: {command}")
            
            # Проверяем метаданные команды мыши
            metadata = sample_data.get('metadata', {})
            screen_id_from_db = metadata.get('screen_id')

            # Проверяем соответствие текущего экрана экрану из базы
            self.wait_screen(screen_id_from_db)  # Ожидаем переход к нужному экрану

            # Выполняем команду мыши
            self._execute_mouse_command(command_action, command, metadata)

            return
        
        # Если мы дошли до сюда, значит команда неизвестного типа
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
            # Учитываем, что после закрывающей скобки могут быть любые символы (включая подчеркивания)
            match = re.search(r'kbd_click_\(([^)]+)\).*', command)
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
            # Учитываем, что после закрывающей скобки могут быть любые символы
            match = re.search(r'kbd_combo_\(([^)]+)\).*', command)
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
    
    def _execute_mouse_command(self, action: str, command: str, metadata: Dict[str, Any]) -> None:
        """
        Выполняет команды мыши.
        
        Параметры:
            action (str): Действие мыши (click, dblclick, scroll).
            command (str): Полная строка команды.
            metadata (Dict[str, Any]): Данные образца.
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
            x, y = get_xy(self.manager, metadata)
            
            # Плавно перемещаем мышь с помощью PyAutoGUI
            pyautogui.moveTo(x, y, duration=settings.PLAYER_MOUSE_MOVE_DURATION)
            
            # Небольшая задержка перед кликом
            time.sleep(settings.PLAYER_MOUSE_MOVE_DELAY)
            
            # Выполняем клик
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
            x, y = get_xy(self.manager, metadata)
            
            # Плавно перемещаем мышь с помощью PyAutoGUI
            pyautogui.moveTo(x, y, duration=settings.PLAYER_MOUSE_MOVE_DURATION)
            
            # Небольшая задержка перед кликом
            time.sleep(settings.PLAYER_MOUSE_MOVE_DELAY)
            
            # Выполняем двойной клик
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
            
            # Определяем координаты для клика
            x, y = get_xy(self.manager, metadata)
            
            # Плавно перемещаем мышь с помощью PyAutoGUI
            pyautogui.moveTo(x, y, duration=settings.PLAYER_MOUSE_MOVE_DURATION)
            
            # Небольшая задержка перед прокруткой
            time.sleep(settings.PLAYER_MOUSE_MOVE_DELAY)
            
            # Выполняем прокрутку
            self.mouse.scroll(dx, dy)

    def play_all(self, commands: List[str]) -> None:
        """
        Воспроизводит список команд последовательно.
        
        Параметры:
            commands (List[str]): Список команд для выполнения.
            
        Вызывает:
            PlaybackError: Если произошла ошибка при воспроизведении команд.
        """
        if not commands:
            print("Список команд пуст")
            return
            
        print(f"Воспроизведение списка из {len(commands)} команд")
        time.sleep(1)

        # Выполняем команды последовательно
        for i, command in enumerate(commands):
            try:
                # Пауза между командами для стабильности работы
                if i > 0:
                    time.sleep(settings.PLAYER_COMMAND_DELAY)
                    
                # Проверка структуры команды, чтобы избежать ошибок при разборе
                if '_' not in command:
                    raise InvalidCommandError(f"Некорректный формат команды: {command}")

                # Выполнение команды
                self.play_one(command)
                
                # print(f"Команда {i+1} успешно выполнена")
                if self.manager.stop_flag:
                    # Установлен флаг сигнализирующий о принудительной остановке выполнения
                    break

            except ScreenMismatchError as e:
                # Выводим ошибку, но продолжаем выполнение следующих команд
                print(f"Ошибка при выполнении команды {i+1} ({command}): {str(e)}")
                # Небольшая пауза перед следующей командой после ошибки
                time.sleep(1.0)
                
            except CommandNotFoundError as e:
                # Выводим ошибку, но продолжаем выполнение следующих команд
                print(f"Ошибка при выполнении команды {i+1} ({command}): {str(e)}")
                # Небольшая пауза перед следующей командой после ошибки
                time.sleep(1.0)
                
            except InvalidCommandError as e:
                # Выводим ошибку, но продолжаем выполнение следующих команд
                print(f"Ошибка при выполнении команды {i+1} ({command}): {str(e)}")
                # Небольшая пауза перед следующей командой после ошибки
                time.sleep(1.0)
                
            except Exception as e:
                # Выводим ошибку, но продолжаем выполнение следующих команд
                print(f"Неожиданная ошибка при выполнении команды {i+1} ({command}): {str(e)}")
                # Небольшая пауза перед следующей командой после ошибки
                time.sleep(1.0)
                
        print(f"Воспроизведение команд завершено")


# Заглушка для менеджера при автономном запуске
class MockManager:
    """Заглушка для менеджера при автономном запуске модуля."""
    
    def __init__(self):
        """Инициализация заглушки менеджера."""
        self.is_standalone = True
        self.screen_id = 'test_screen_id'
        self.screenshot = None
        # Нет необходимости в создании MockChroma, так как теперь используется прямой доступ к chroma_db


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
        
        # Используем текущий screen_id для всех команд клавиатуры
        screen_id = 'test_screen_id'  # ID экрана в тестовом режиме
        
        # Формируем команды для ввода слова "Привет" с использованием правильного формата
        commands = [
            # Для заглавной буквы П - комбинация Shift+п
            f"kbd_combo_(shift п)_{screen_id}",
            f"kbd_click_(р)_{screen_id}",
            f"kbd_click_(и)_{screen_id}",
            f"kbd_click_(в)_{screen_id}",
            f"kbd_click_(е)_{screen_id}",
            f"kbd_click_(т)_{screen_id}",
        ]
        
        print(f"Выполнение команд для набора слова 'Привет'...")
        player.play_all(commands)
        
        print("Тестирование завершено!")
        
    except Exception as e:
        print(f"Ошибка при тестировании: {str(e)}")
        