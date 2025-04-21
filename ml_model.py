#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль машинного обучения (ml_model.py)
=======================================

Модуль предоставляет класс для создания и использования модели машинного обучения,
предназначенной для получения векторных представлений (эмбеддингов) изображений.
Использует архитектуру MobileNetV2 без верхних слоев для эффективного извлечения признаков.

Класс Model инкапсулирует функциональность по подготовке и обработке изображений,
а также генерации векторных представлений для последующего использования в системе
сравнения и поиска изображений.

Модуль оптимизирован для многопроцессной работы: 
- Модель TensorFlow создается только в главном процессе во время импорта
- Дочерние процессы используют ленивую инициализацию
"""

import numpy as np
import os
import logging
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import time
import multiprocessing
from typing import List, Union, Tuple

# Отключаем предупреждения TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Устанавливаем параметры для oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Отключаем oneDNN для предотвращения предупреждений

# Глобальная переменная для хранения модели
_global_model_instance = None
_is_initialized = False  # Флаг, указывающий, инициализирована ли модель

# Определяем, является ли текущий процесс главным
_is_main_process = multiprocessing.current_process().name == 'MainProcess'

class Model:
    """
    Класс для создания и использования модели машинного обучения для эмбеддинга изображений.
    
    Предоставляет методы для преобразования изображений в векторные представления
    с использованием предобученной модели MobileNetV2.
    
    Реализует паттерн Singleton с ленивой инициализацией для оптимизации многопроцессной работы.
    Модель TensorFlow создается только один раз в главном процессе, а дочерние процессы
    используют ленивую инициализацию при первом обращении к модели.
    """
    
    def __new__(cls):
        """
        Создает новый экземпляр класса или возвращает существующий (синглтон).
        
        Returns:
            Model: Единственный экземпляр класса Model.
        """
        global _global_model_instance
        if _global_model_instance is None:
            process_type = "главном" if _is_main_process else "дочернем"
            print(f"[ML] Создание экземпляра модели в {process_type} процессе")
            _global_model_instance = super(Model, cls).__new__(cls)
            _global_model_instance._model = None
            _global_model_instance._initialized = False
        return _global_model_instance
    
    def __init__(self) -> None:
        """
        Подготовка к инициализации модели MobileNetV2.
        
        Фактическая инициализация модели происходит только при первом вызове get_embedding или get_embedding_with_time.
        В главном процессе модель инициализируется при импорте, в дочерних - по требованию.
        """
        global _is_initialized
        
        # Устанавливаем базовые параметры если класс еще не инициализирован
        if not self._initialized:
            # Фиксированный размер входного изображения для MobileNetV2
            self.input_size = (224, 224)
            self._initialized = True
            print(f"[ML] Объект модели создан в процессе {multiprocessing.current_process().name}")
            
            # В главном процессе сразу инициализируем модель
            if _is_main_process and not _is_initialized:
                self._initialize_model()
                _is_initialized = True
    
    def _initialize_model(self) -> None:
        """
        Отложенная инициализация модели при первом использовании.
        
        В главном процессе инициализация происходит при импорте.
        В дочерних процессах - при первом обращении к модели.
        """
        global _is_initialized
        
        # Если модель уже инициализирована, просто возвращаемся
        if self._model is not None:
            return
            
        process_type = "главном" if _is_main_process else "дочернем"
        print(f"[ML] Инициализация модели MobileNetV2 в {process_type} процессе")
        
        # Создаем модель с отключенным выводом
        with tf.keras.utils.CustomObjectScope({}):
            self._model = MobileNetV2(
                weights="imagenet",           # Используем предобученные веса
                include_top=False,            # Не включаем верхние слои классификации
                pooling="avg",                # Используем усреднение для получения одномерного вектора
                input_shape=(*self.input_size, 3)  # Размер входного изображения (224x224x3)
            )
        
        # Компилируем модель для оптимизации производительности
        self._model.compile(optimizer='adam', loss='mse')
        
        # Прогреваем модель (warm-up) для инициализации весов
        dummy_input = np.zeros((1, *self.input_size, 3), dtype=np.float32)
        _ = self._model.predict(dummy_input, verbose=0)
        
        _is_initialized = True
        print(f"[ML] Модель инициализирована в процессе {multiprocessing.current_process().name}")
    
    @property
    def model(self):
        """
        Доступ к модели с ленивой инициализацией.
        
        Returns:
            tensorflow.keras.Model: Инициализированная модель TensorFlow.
        """
        if self._model is None:
            self._initialize_model()
        return self._model
        
    def get_embedding(self, image: np.ndarray) -> List[float]:
        """
        Выполняет эмбеддинг изображения, возвращая его векторное представление.
        
        Параметры:
            image (np.ndarray): Изображение в формате numpy array.
            
        Возвращает:
            List[float]: Векторное представление изображения в виде списка чисел с плавающей точкой.
        """
        # Проверяем количество каналов и преобразуем из BGRA в BGR если нужно
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Изображение имеет 4 канала (BGRA), оставляем только 3 (BGR)
            image = image[:, :, :3]
        
        # Изменяем размер изображения до требуемого для модели
        processed_img = cv2.resize(image, self.input_size)
        
        # Предобработка изображения для модели MobileNetV2
        processed_img = preprocess_input(processed_img)
        
        # Добавляем размерность батча (1 изображение)
        img_batch = np.expand_dims(processed_img, axis=0)
        
        # Измеряем время выполнения
        start_time = time.time()  # Начало замера времени
        
        # Получаем эмбеддинг с помощью модели (отключаем вывод прогресса)
        with tf.device('/CPU:0'):  # Принудительное использование CPU для стабильности
            embedding = self.model.predict(img_batch, verbose=0)[0]
        
        end_time = time.time()  # Конец замера времени
        duration = end_time - start_time
        print(f"[ML] Время создания вектора: {duration:.6f} секунд в процессе {multiprocessing.current_process().name}")
        
        # Преобразуем numpy array в список и возвращаем
        return embedding.tolist()
        
    def get_embedding_with_time(self, image: np.ndarray) -> Tuple[List[float], float]:
        """
        Выполняет эмбеддинг изображения и возвращает векторное представление вместе с временем выполнения.
        
        Параметры:
            image (np.ndarray): Изображение в формате numpy array.
            
        Возвращает:
            Tuple[List[float], float]: Кортеж из векторного представления изображения и времени (в секундах),
                                      затраченного на создание вектора.
        """
        # Проверяем количество каналов и преобразуем из BGRA в BGR если нужно
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Изображение имеет 4 канала (BGRA), оставляем только 3 (BGR)
            image = image[:, :, :3]
        
        # Изменяем размер изображения до требуемого для модели
        processed_img = cv2.resize(image, self.input_size)
        
        # Предобработка изображения для модели MobileNetV2
        processed_img = preprocess_input(processed_img)
        
        # Добавляем размерность батча (1 изображение)
        img_batch = np.expand_dims(processed_img, axis=0)
        
        # Измеряем время выполнения
        start_time = time.time()  # Начало замера времени
        
        # Получаем эмбеддинг с помощью модели (отключаем вывод прогресса)
        with tf.device('/CPU:0'):  # Принудительное использование CPU для стабильности
            embedding = self.model.predict(img_batch, verbose=0)[0]
        
        end_time = time.time()  # Конец замера времени
        duration = end_time - start_time
        
        # Преобразуем numpy array в список и возвращаем вместе с временем выполнения
        return embedding.tolist(), duration

# Создаем экземпляр модели, который инициализируется при импорте в главном процессе
# и при первом использовании в дочерних процессах
model = Model()

# Определение для тестирования
if __name__ == "__main__":
    print("[ML] Тестирование модуля ml_model в автономном режиме")
    
    # Создаем тестовое изображение
    test_image = np.random.rand(100, 100, 3) * 255
    test_image = test_image.astype(np.uint8)
    
    # Получаем эмбеддинг и время
    embedding, duration = model.get_embedding_with_time(test_image)
    
    print(f"[ML] Получен вектор размерности {len(embedding)}")
    print(f"[ML] Время выполнения: {duration:.6f} секунд")