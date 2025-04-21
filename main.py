#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль main
===========

Предоставляет функции для запуска сервисов пакета и обработки команд.
Может использоваться как библиотека или запускаться из командной строки.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional, Union, Any

from manager import Manager
from player import Player
from input_tracker import InputTracker
import settings
from exceptions import (
    BaseAppError, 
    InvalidCommandError, 
    PlaybackError,
    InputError
)

def run(command: str) -> None:
    """
    Обрабатывает и выполняет команду.
    
    Парсит команду, определяет тип действия и передает на выполнение нужному исполнителю.
    
    Параметры:
        command (str): Строка команды для выполнения.
    
    Возвращает:
        None
    """
    if not command:
        raise InvalidCommandError("Пустая команда")
    
    parts = command.strip().split('_')
    if not parts:
        raise InvalidCommandError(f"Некорректный формат команды: {command}")
    
    action = parts[0].lower()
    
    if action == "play":
        # Выполнение скрипта команд
        filename = parts[1] if len(parts) > 1 else None
        play(filename)
    elif action == "record":
        # Запись скрипта команд
        filename = parts[1] if len(parts) > 1 else None
        record(filename)
    elif action in ["kbd", "mouse"]:
        # Выполнение одиночной команды
        manager = Manager()
        try:
            manager.start()
            player_instance = Player(manager)
            player_instance.play_one(command)
        finally:
            manager.stop()
    else:
        raise InvalidCommandError(f"Неизвестная команда: {command}")

def play(filename: Optional[str] = None) -> None:
    """
    Воспроизводит скрипт команд из файла.
    
    Параметры:
        filename (Optional[str]): Имя файла без расширения. 
                                 Если None, используется файл по умолчанию.
    
    Возвращает:
        None
    """
    # Определение имени файла
    if filename is None:
        target_file = settings.DEFAULT_COMMANDS_FILE
    else:
        target_file = f"{filename}.json" if not filename.endswith(".json") else filename
    
    # Проверка существования файла
    if not os.path.exists(target_file):
        raise PlaybackError(f"Файл команд не найден: {target_file}")
    
    # Загрузка команд из файла
    try:
        with open(target_file, 'r', encoding='utf-8') as file:
            commands = json.load(file)
    except json.JSONDecodeError:
        raise PlaybackError(f"Ошибка чтения JSON из файла: {target_file}")
    except Exception as e:
        raise PlaybackError(f"Ошибка при чтении файла {target_file}: {str(e)}")
    
    # Создаем Manager и Player для воспроизведения команд
    manager = Manager()
    try:
        manager.start()
        player_instance = Player(manager)
        player_instance.play_all(commands)
    finally:
        manager.stop()

def record(filename: Optional[str] = None) -> None:
    """
    Записывает действия пользователя в файл скрипта.
    
    Параметры:
        filename (Optional[str]): Имя файла без расширения. 
                                 Если None, используется файл по умолчанию.
    
    Возвращает:
        None
    """
    # Определение имени файла
    if filename is None:
        target_file = settings.DEFAULT_COMMANDS_FILE
    else:
        target_file = f"{filename}.json" if not filename.endswith(".json") else filename
    
    print(f"Начинаю запись действий в файл: {target_file}")
    print("Для завершения записи нажмите дважды ESC")
    
    # Создаем Manager для записи действий
    manager = Manager()
    try:
        # Запускаем Manager
        manager.start()
        
        # Создание и запуск трекера ввода с передачей объекта Manager
        input_tracker = InputTracker(manager)
        commands = input_tracker.start()
        
        # Сохранение списка команд в файл
        with open(target_file, 'w', encoding='utf-8') as file:
            json.dump(commands, file, ensure_ascii=False, indent=2)
        
        print(f"Запись завершена. Сохранено {len(commands)} команд в файл {target_file}")
    except InputError as e:
        raise PlaybackError(f"Ошибка при записи действий: {str(e)}")
    except Exception as e:
        raise PlaybackError(f"Неожиданная ошибка при записи: {str(e)}")
    finally:
        # Останавливаем Manager
        manager.stop()

def cli_main() -> None:
    """
    Точка входа для запуска из командной строки.
    
    Инициализирует Manager, обрабатывает аргументы командной строки
    и выполняет соответствующие действия.
    
    Возвращает:
        None
    """
    parser = argparse.ArgumentParser(description="Управление автоматизацией действий пользователя")
    parser.add_argument('command', nargs='?', default='', help='Команда для выполнения')
    parser.add_argument('params', nargs='*', help='Параметры команды')
    
    args = parser.parse_args()
    
    # Получаем полную команду из аргументов
    full_command = args.command
    if args.params:
        full_command += "_" + "_".join(args.params)
    
    manager = Manager()
    
    try:
        # Запускаем Manager
        manager.start()
        
        # Выполняем команду
        if full_command:
            run(full_command)
        else:
            print("Укажите команду для выполнения")
            print("Примеры команд:")
            print("  play [имя_файла] - воспроизвести скрипт")
            print("  record [имя_файла] - записать скрипт")
            print("  kbd_click_(key)_event_id - нажать клавишу")
            print("  mouse_click_left_event_id - клик мышью")
    except BaseAppError as e:
        from exceptions import handle_exception
        handle_exception(e)
    except Exception as e:
        print(f"Неизвестная ошибка: {str(e)}")
    finally:
        # Останавливаем Manager
        manager.stop()

if __name__ == "__main__":
    cli_main() 