#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль input_tracker
=====================

Модуль предназначен для отслеживания и регистрации событий ввода пользователя,
включая нажатия клавиш, клики мыши и прокрутку колесика мыши.
"""

import time
import os
import json
import sys
from typing import List, Dict, Any, Tuple, Set, Optional, Callable, Union
from pynput import keyboard, mouse
from collections import defaultdict

# Импорт из других модулей проекта
import settings
from exceptions import KeyboardError, MouseError, InputError
from images import set_sample


class InputTracker:
    """
    Класс для отслеживания и регистрации событий ввода пользователя.
    
    Отслеживает:
    - Клики мыши (левой, правой и средней кнопками)
    - Прокрутку колесика мыши
    - Нажатия клавиш клавиатуры
    - Комбинации клавиш
    
    Выход из режима отслеживания осуществляется двойным нажатием клавиши ESC.
    """
    
    def __init__(self, manager=None):
        """
        Инициализация трекера ввода.
        
        Параметры:
            manager: Объект Manager для доступа к компонентам системы
        """
        # Сохраняем объект Manager
        self.manager = manager
        
        # Флаг выхода из цикла отслеживания
        self.exit_flag = False
        
        # Время последнего нажатия ESC для определения двойного нажатия
        self.last_esc_press_time = 0
        
        # Для отложенной обработки ESC
        self.pending_esc = False
        self.pending_esc_time = 0
        self.esc_timeout = 0.5  # 500 мс для определения двойного нажатия
        
        # Для обработки двойного клика мыши
        self.last_click_time = 0
        self.last_click_position = (0, 0)
        self.last_click_button = None
        self.dblclick_timeout = 0.3  # 300 мс для определения двойного клика
        self.pending_click = False
        self.pending_click_time = 0
        self.pending_click_button = None
        self.pending_click_position = (0, 0)
        self.pending_click_sample_id = None
        
        # Список записанных команд
        self.commands = []
        
        # Слушатели клавиатуры и мыши
        self.keyboard_listener = None
        self.mouse_listener = None
        
        # Для комбинаций клавиш
        self.pressed_keys = set()
        self.is_combo_active = False
        self.combo_keys = set()
        # Множество клавиш, которые уже были обработаны как часть комбинации
        self.processed_combo_keys = set()
        
        # Для именования специальных клавиш
        self.special_keys = {
            keyboard.Key.ctrl: 'ctrl',
            keyboard.Key.ctrl_l: 'ctrl',
            keyboard.Key.ctrl_r: 'ctrl',
            keyboard.Key.alt: 'alt',
            keyboard.Key.alt_l: 'alt',
            keyboard.Key.alt_r: 'alt',
            keyboard.Key.shift: 'shift',
            keyboard.Key.shift_l: 'shift',
            keyboard.Key.shift_r: 'shift',
            keyboard.Key.cmd: 'cmd',
            keyboard.Key.cmd_l: 'cmd',
            keyboard.Key.cmd_r: 'cmd',
            keyboard.Key.enter: 'enter',
            keyboard.Key.backspace: 'backspace',
            keyboard.Key.tab: 'tab',
            keyboard.Key.space: 'space',
            keyboard.Key.esc: 'esc',
            keyboard.Key.caps_lock: 'caps_lock',
            keyboard.Key.f1: 'f1',
            keyboard.Key.f2: 'f2',
            keyboard.Key.f3: 'f3',
            keyboard.Key.f4: 'f4',
            keyboard.Key.f5: 'f5',
            keyboard.Key.f6: 'f6',
            keyboard.Key.f7: 'f7',
            keyboard.Key.f8: 'f8',
            keyboard.Key.f9: 'f9',
            keyboard.Key.f10: 'f10',
            keyboard.Key.f11: 'f11',
            keyboard.Key.f12: 'f12',
            keyboard.Key.home: 'home',
            keyboard.Key.end: 'end',
            keyboard.Key.page_up: 'page_up',
            keyboard.Key.page_down: 'page_down',
            keyboard.Key.insert: 'insert',
            keyboard.Key.delete: 'delete',
            keyboard.Key.left: 'left',
            keyboard.Key.right: 'right',
            keyboard.Key.up: 'up',
            keyboard.Key.down: 'down',
            keyboard.Key.num_lock: 'num_lock',
            keyboard.Key.print_screen: 'print_screen',
            keyboard.Key.scroll_lock: 'scroll_lock',
            keyboard.Key.pause: 'pause',
            keyboard.Key.menu: 'menu'
        }
        
        # Для отслеживания нажатых модификаторов
        self.ctrl_pressed = False
        self.alt_pressed = False
        self.shift_pressed = False
        
        # Словарь для преобразования ctrl+клавиша
        self.ctrl_key_map = {
            '\x01': 'a',
            '\x02': 'b',
            '\x03': 'c',
            '\x04': 'd',
            '\x05': 'e',
            '\x06': 'f',
            '\x07': 'g',
            '\x08': 'h',
            '\t': 'tab',
            '\n': 'j',
            '\x0b': 'k',
            '\x0c': 'l',
            '\r': 'm',
            '\x0e': 'n',
            '\x0f': 'o',
            '\x10': 'p',
            '\x11': 'q',
            '\x12': 'r',
            '\x13': 's',
            '\x14': 't',
            '\x15': 'u',
            '\x16': 'v',
            '\x17': 'w',
            '\x18': 'x',
            '\x19': 'y',
            '\x1a': 'z'
        }
        
        # Словарь соответствия русских и английских букв
        self.ru_en_map = {
            'й': 'q', 'ц': 'w', 'у': 'e', 'к': 'r', 'е': 't', 'н': 'y', 'г': 'u', 
            'ш': 'i', 'щ': 'o', 'з': 'p', 'х': '[', 'ъ': ']', 'ф': 'a', 'ы': 's', 
            'в': 'd', 'а': 'f', 'п': 'g', 'р': 'h', 'о': 'j', 'л': 'k', 'д': 'l', 
            'ж': ';', 'э': "'", 'я': 'z', 'ч': 'x', 'с': 'c', 'м': 'v', 'и': 'b', 
            'т': 'n', 'ь': 'm', 'б': ',', 'ю': '.', '.': '/', ',': '.', 
            # Заглавные буквы
            'Й': 'Q', 'Ц': 'W', 'У': 'E', 'К': 'R', 'Е': 'T', 'Н': 'Y', 'Г': 'U',
            'Ш': 'I', 'Щ': 'O', 'З': 'P', 'Х': '{', 'Ъ': '}', 'Ф': 'A', 'Ы': 'S',
            'В': 'D', 'А': 'F', 'П': 'G', 'Р': 'H', 'О': 'J', 'Л': 'K', 'Д': 'L',
            'Ж': ':', 'Э': '"', 'Я': 'Z', 'Ч': 'X', 'С': 'C', 'М': 'V', 'И': 'B',
            'Т': 'N', 'Ь': 'M', 'Б': '<', 'Ю': '>', '?': '?'
        }
        
        # Словарь обратного соответствия для определения, является ли символ русским
        self.en_ru_map = {v: k for k, v in self.ru_en_map.items()}
    
    def start(self) -> List[str]:
        """
        Запуск отслеживания ввода пользователя.
        
        Returns:
            List[str]: Список записанных команд
        """
        try:
            # Инициализация слушателей
            self.keyboard_listener = keyboard.Listener(
                on_press=self.on_key_press,
                on_release=self.on_key_release
            )
            
            self.mouse_listener = mouse.Listener(
                on_click=self.on_mouse_click,
                on_scroll=self.on_mouse_scroll
            )
            
            # Запуск слушателей
            self.keyboard_listener.start()
            self.mouse_listener.start()
            
            print("Отслеживание ввода запущено. Для выхода нажмите ESC дважды.")
            
            # Основной цикл отслеживания
            while not self.exit_flag:
                time.sleep(0.05)  # Более частая проверка
                
                # Обработка отложенного ESC в основном цикле
                if self.pending_esc and (time.time() - self.pending_esc_time >= self.esc_timeout):
                    # Прошло достаточно времени после нажатия ESC, и не было второго нажатия
                    # Регистрируем одиночный ESC
                    command = f"kbd_click_(esc)_{self.manager.screen_id}"
                    self.commands.append(command)

                    print(f"Нажатие клавиши: {command}")
                    self.pending_esc = False
                
                # Обработка отложенного клика мыши в основном цикле
                if self.pending_click and (time.time() - self.pending_click_time >= self.dblclick_timeout):
                    # Прошло достаточно времени после клика, и не было второго клика

                    # Регистрируем одиночный клик
                    button_type = "left"
                    if self.pending_click_button == mouse.Button.right:
                        button_type = "right"
                    elif self.pending_click_button == mouse.Button.middle:
                        button_type = "middle"
                    
                    # Формируем команду
                    self.manager.add_command(f"mouse_click_{button_type}")  # Обновляем данные о событии в менеджере
                    x, y = self.pending_click_position
                    sample_id = set_sample(self.manager, x, y)

                    command = f"mouse_click_{button_type}_{sample_id}"
                    self.commands.append(command)
                    print(f"Клик мыши ({button_type}) по координатам {self.pending_click_position}: {command}")
                    self.pending_click = False

                    self.manager.screen_update()  # Экран должен обновиться после события

            return self.commands
        except Exception as e:
            raise InputError(f"Ошибка при запуске отслеживания: {str(e)}")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """
        Останавливает отслеживание и освобождает ресурсы.
        """
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        
        if self.mouse_listener:
            self.mouse_listener.stop()
        
        print("Отслеживание ввода остановлено.")
    
    def on_key_press(self, key) -> None:
        """
        Обработчик события нажатия клавиши.
        
        Args:
            key: Объект клавиши pynput
        """
        try:
            # Проверка на двойное нажатие ESC для выхода
            if key == keyboard.Key.esc:
                current_time = time.time()
                # Если уже есть отложенный ESC и второй нажали быстро - это двойное нажатие ESC для выхода
                if self.pending_esc and (current_time - self.last_esc_press_time < self.esc_timeout):
                    # Это двойное нажатие ESC - устанавливаем флаг выхода
                    self.exit_flag = True
                    # Отменяем отложенную регистрацию первого ESC
                    self.pending_esc = False
                    return
                # Сохраняем время для проверки двойного нажатия ESC
                self.last_esc_press_time = current_time
                # Устанавливаем отложенную регистрацию ESC
                self.pending_esc = True
                self.pending_esc_time = current_time
                return  # Не обрабатываем ESC дальше
            
            # Отслеживаем нажатие модификаторов
            if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                self.ctrl_pressed = True
            elif key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
                self.alt_pressed = True
            elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                self.shift_pressed = True
            
            # Добавляем клавишу в множество нажатых клавиш
            self.pressed_keys.add(key)
            
            # Обновляем комбинацию, если уже начата или если нажато 2+ клавиши
            if self.is_combo_active:
                self.combo_keys.add(key)
            elif len(self.pressed_keys) >= 2:
                self.is_combo_active = True
                self.combo_keys = self.pressed_keys.copy()
                self.processed_combo_keys.clear()  # Очищаем множество обработанных клавиш
            
        except Exception as e:
            raise KeyboardError(f"Ошибка при обработке нажатия клавиши: {str(e)}")
    
    def on_key_release(self, key) -> None:
        """
        Обработчик события отпускания клавиши.
        
        Args:
            key: Объект клавиши pynput
        """
        try:
            # Если установлен флаг выхода, пропускаем все клавиши
            if self.exit_flag:
                return
            
            # Проверяем отложенное нажатие ESC
            if key == keyboard.Key.esc:
                # Если это был второй ESC для двойного нажатия (уже обработан в on_key_press),
                # просто пропускаем его
                current_time = time.time()
                if current_time - self.last_esc_press_time < self.esc_timeout:
                    # Это отпускание для двойного нажатия - уже обработано в on_key_press
                    return
                
            # Если клавиша уже была частью обработанной комбинации, пропускаем её
            if key in self.processed_combo_keys:
                self.processed_combo_keys.remove(key)
                if key in self.pressed_keys:
                    self.pressed_keys.remove(key)
                # Обновляем состояние модификаторов, даже если клавиша была обработана
                if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                    self.ctrl_pressed = False
                elif key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
                    self.alt_pressed = False
                elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                    self.shift_pressed = False
                return
                
            # Удаляем клавишу из множества нажатых клавиш
            if key in self.pressed_keys:
                self.pressed_keys.remove(key)
            
            # Если была активна комбинация клавиш и одна из её клавиш отпущена
            if self.is_combo_active: # Важно: убрано условие key in self.combo_keys
                # Убеждаемся, что у нас не пустая комбинация
                if len(self.combo_keys) >= 2:
                    # Собираем все клавиши в комбинации
                    all_combo_keys = list(self.combo_keys)
                    
                    # Выделяем модификаторы
                    modifiers = []
                    regular_keys = []
                    
                    # Добавляем модификаторы из состояния флагов
                    if self.ctrl_pressed:
                        modifiers.append('ctrl')
                    if self.alt_pressed:
                        modifiers.append('alt')
                    if self.shift_pressed:
                        modifiers.append('shift')
                    
                    # Затем добавляем все остальные клавиши
                    for k in all_combo_keys:
                        if k not in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
                                   keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r,
                                   keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r,
                                   keyboard.Key.esc):  # Исключаем ESC из комбинаций
                            # Обрабатываем обычные клавиши
                            name = self.get_key_name(k)
                            
                            # Преобразуем символы в зависимости от нажатых модификаторов
                            if name and len(name) == 1:
                                # Если это управляющий символ и нажат Ctrl
                                if self.ctrl_pressed and isinstance(k, keyboard.KeyCode) and k.char in self.ctrl_key_map:
                                    name = self.ctrl_key_map[k.char]
                            
                            if name and name not in regular_keys:
                                regular_keys.append(name)
                    
                    # Добавляем текущую клавишу, если она не модификатор и не уже в regular_keys
                    if key not in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
                                 keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r,
                                 keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r,
                                 keyboard.Key.esc):  # Исключаем ESC из комбинаций
                        name = self.get_key_name(key)
                        if name and name not in regular_keys:
                            regular_keys.append(name)
                    
                    # Если есть хотя бы один модификатор и хотя бы одна обычная клавиша
                    # или есть как минимум два модификатора
                    if (modifiers and regular_keys) or len(modifiers) >= 2:
                        # Формируем конечный список клавиш: сначала модификаторы, потом обычные клавиши
                        combo_keys_names = modifiers + regular_keys
                        
                        # Формируем команду
                        combo_str = " ".join(combo_keys_names).strip()
                        command = f"kbd_combo_({combo_str})_{self.manager.screen_id}"
                        self.commands.append(command)
                        print(f"Комбинация клавиш: {command}")
                        
                        # Помечаем все клавиши комбинации как обработанные, кроме текущей
                        self.processed_combo_keys = self.combo_keys.copy()
                        self.processed_combo_keys.discard(key)
                    else:
                        # Если нет комбинации модификаторов с обычными клавишами,
                        # обрабатываем как одиночное нажатие
                        if not (key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
                                      keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r,
                                      keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r,
                                      keyboard.Key.esc)):  # Исключаем ESC
                            key_name = self.get_key_name(key)
                            if key_name and key_name.strip():
                                command = f"kbd_click_({key_name})_{self.manager.screen_id}"
                                self.commands.append(command)
                                print(f"Нажатие клавиши: {command}")
                else:
                    # Если в комбинации осталась только одна клавиша, обрабатываем как одиночное нажатие
                    if not (key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
                                  keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r,
                                  keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r,
                                  keyboard.Key.esc)):  # Исключаем ESC
                        key_name = self.get_key_name(key)
                        if key_name and key_name.strip():
                            command = f"kbd_click_({key_name})_{self.manager.screen_id}"
                            self.commands.append(command)
                            print(f"Нажатие клавиши: {command}")
                
                # Обновляем состояние модификаторов ПОСЛЕ обработки комбинации
                if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                    self.ctrl_pressed = False
                elif key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
                    self.alt_pressed = False
                elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                    self.shift_pressed = False
                
                # Сбрасываем состояние комбинации
                self.is_combo_active = False
                self.combo_keys.clear()
            
            # Если не было комбинации, записываем одиночное нажатие
            elif not self.is_combo_active:
                # Обрабатываем не-модификаторы и не-ESC
                if not (key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
                              keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r,
                              keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r,
                              keyboard.Key.esc)):  # Исключаем ESC
                    key_name = self.get_key_name(key)
                    # Проверяем, что имя клавиши не пустое
                    if key_name and key_name.strip():
                        command = f"kbd_click_({key_name})_{self.manager.screen_id}"
                        self.commands.append(command)
                        print(f"Нажатие клавиши: {command}")
                
                # Обновляем состояние модификаторов
                if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                    self.ctrl_pressed = False
                elif key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
                    self.alt_pressed = False
                elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                    self.shift_pressed = False

        except Exception as e:
            raise KeyboardError(f"Ошибка при обработке отпускания клавиши: {str(e)}")

        self.manager.screen_update()  # Экран должен обновиться после события

    def on_mouse_click(self, x, y, button, pressed) -> None:
        """
        Обработчик событий мыши (клики).
        
        Args:
            x (int): Координата X курсора
            y (int): Координата Y курсора
            button: Кнопка мыши (объект pynput)
            pressed (bool): True если нажата, False если отпущена
        """
        try:
            # Обрабатываем только отпускание кнопки (завершение клика)
            if not pressed:
                current_time = time.time()
                current_position = (x, y)
                
                # Определяем тип кнопки мыши
                button_type = "left"
                if button == mouse.Button.right:
                    button_type = "right"
                elif button == mouse.Button.middle:
                    button_type = "middle"
                
                # Если у нас есть отложенный клик, проверяем, можно ли его считать частью двойного клика
                if self.pending_click:
                    # Отменяем обработку отложенного клика, т.к. получен второй клик
                    self.pending_click = False
                    
                    # Проверяем, что это тот же тип кнопки и примерно та же позиция (с допуском в 5 пикселей)
                    if (button == self.pending_click_button and
                        abs(current_position[0] - self.pending_click_position[0]) <= 5 and
                        abs(current_position[1] - self.pending_click_position[1]) <= 5 and
                        current_time - self.pending_click_time < self.dblclick_timeout):

                        # Получаем образец изображения только один раз при двойном клике
                        # Экран обновлять не нужно, он уже был обновлен при первом клике
                        self.manager.add_command(f"mouse_dblclick_{button_type}")  # Обновляем данные о событии в менеджере
                        sample_id = set_sample(self.manager, x, y)

                        command = f"mouse_dblclick_{button_type}_{sample_id}"
                        self.commands.append(command)
                        print(f"Двойной клик мыши ({button_type}) по координатам ({x}, {y}): {command}")
                        
                        # Сбрасываем информацию о последнем клике
                        self.last_click_time = 0
                        self.last_click_position = (0, 0)
                        self.last_click_button = None

                        self.manager.screen_update()  # Экран должен обновиться после события
                        return

                # Сохраняем информацию о текущем клике для возможного определения двойного клика
                self.pending_click = True
                self.pending_click_time = current_time
                self.pending_click_button = button
                self.pending_click_position = current_position
                self.pending_click_sample_id = None  # Образец будет получен позже
                
                # Не добавляем команду сразу - отложим до проверки на двойной клик
                
        except Exception as e:
            raise MouseError(f"Ошибка при обработке клика мыши: {str(e)}")
    
    def on_mouse_scroll(self, x, y, dx, dy) -> None:
        """
        Обработчик событий прокрутки колеса мыши.
        
        Args:
            x (int): Координата X курсора
            y (int): Координата Y курсора
            dx (int): Горизонтальная прокрутка
            dy (int): Вертикальная прокрутка
        """
        self.manager.screen_update()  # Экран должен обновиться после события

        try:
            # Получаем образец изображения в месте прокрутки
            self.manager.add_command(f"mouse_scroll_{dx},{dy}")  # Обновляем данные о событии в менеджере
            sample_id = set_sample(self.manager, x, y)
            
            # Формируем команду
            command = f"mouse_scroll_{dx},{dy}_{sample_id}"
            self.commands.append(command)
            print(f"Прокрутка мыши по координатам ({x}, {y}), смещение ({dx}, {dy}): {command}")
            
        except Exception as e:
            raise MouseError(f"Ошибка при обработке прокрутки мыши: {str(e)}")

        self.manager.screen_update()  # Экран должен обновиться после события

    def get_key_name(self, key) -> str:
        """
        Преобразует объект клавиши в строковое представление.
        
        Args:
            key: Объект клавиши pynput
        
        Returns:
            str: Строковое представление клавиши
        """
        # Проверяем, является ли клавиша специальной
        if key in self.special_keys:
            return self.special_keys[key]
        
        # Для обычных символьных клавиш
        try:
            char = key.char
            if char is None:
                return ""
                
            # Проверяем, является ли это комбинацией с Ctrl
            if self.ctrl_pressed and char in self.ctrl_key_map:
                return self.ctrl_key_map[char]
            
            # Оставляем символ как есть, сохраняя русские буквы
            return char
            
        except AttributeError:
            # Если не удалось получить символ, используем строковое представление
            return str(key).replace("'", "")


def main():
    """
    Точка входа для запуска из терминала.
    Записывает события ввода и выводит их в терминал.
    """
    try:
        from manager import Manager
        
        # Инициализируем менеджер
        manager = Manager(action_name="record")
        
        try:
            # Создаем и запускаем трекер ввода
            tracker = InputTracker(manager)
            commands = tracker.start()
            
            # Выводим список собранных команд
            print("\nСписок записанных команд:")
            for i, cmd in enumerate(commands, 1):
                print(f"{i}. {cmd}")
                
            # Сохраняем команды в файл по умолчанию
            output_file = settings.DEFAULT_COMMANDS_FILE
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(commands, file, ensure_ascii=False, indent=2)
            print(f"Команды сохранены в файл: {output_file}")
            
        finally:
            # Останавливаем менеджер
            manager.stop()
                   
    except Exception as e:
        print(f"Ошибка при выполнении трекера ввода: {str(e)}")


# Определяем константу для проверки прямого запуска модуля
DIRECT_RUN = __name__ == "__main__"

if DIRECT_RUN:
    main() 