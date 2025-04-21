#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для обработки исключений
===============================

Модуль предоставляет классы исключений для различных компонентов системы.
Используется для унифицированной обработки ошибок во всем приложении.
"""

class BaseAppError(Exception):
    """Базовый класс для всех исключений приложения."""
    pass

# Ошибки базы данных
class ChromaDBError(BaseAppError):
    """Ошибка при работе с ChromaDB."""
    pass

class CollectionNotFoundError(ChromaDBError):
    """Ошибка: коллекция не найдена."""
    pass

class DocumentNotFoundError(ChromaDBError):
    """Ошибка: документ не найден."""
    pass

class DuplicateIDError(ChromaDBError):
    """Ошибка: попытка добавить документ с существующим ID."""
    pass

# Ошибки воспроизведения
class ScreenMismatchError(BaseAppError):
    """Ошибка: несоответствие текущего экрана требуемому."""
    pass

class CommandNotFoundError(BaseAppError):
    """Ошибка: команда не найдена в базе данных."""
    pass

class InvalidCommandError(BaseAppError):
    """Ошибка: некорректный формат команды."""
    pass

class PlaybackError(BaseAppError):
    """Общая ошибка при воспроизведении."""
    pass

# Ошибки мониторинга
class ScreenCaptureError(BaseAppError):
    """Ошибка при захвате экрана."""
    pass

class MemoryError(BaseAppError):
    """Ошибка работы с shared memory."""
    pass

class MonitorError(BaseAppError):
    """Общая ошибка мониторинга."""
    pass

# Ошибки модели
class ModelError(BaseAppError):
    """Ошибка при работе с ML моделью."""
    pass

class EmbeddingError(BaseAppError):
    """Ошибка при генерации эмбеддингов."""
    pass

# Ошибки настроек
class SettingsError(BaseAppError):
    """Ошибка при загрузке/применении настроек."""
    pass

class InvalidParameterError(BaseAppError):
    """Ошибка: некорректное значение параметра."""
    pass

# Ошибки ввода
class InputError(BaseAppError):
    """Общая ошибка при обработке ввода."""
    pass

class MouseError(InputError):
    """Ошибка при работе с мышью."""
    pass

class KeyboardError(InputError):
    """Ошибка при работе с клавиатурой."""
    pass

def handle_exception(e: BaseAppError) -> None:
    """
    Обрабатывает исключения и выводит сообщение об ошибке.
    
    Параметры:
        e (BaseAppError): Исключение для обработки.
    """
    print(f"Произошла ошибка: {str(e)}")