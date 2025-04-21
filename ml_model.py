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
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import time  # Добавляем импорт модуля time для замера времени
from typing import List, Union, Tuple


class Model:
    """
    Класс для создания и использования модели машинного обучения для эмбеддинга изображений.
    
    Предоставляет методы для преобразования изображений в векторные представления
    с использованием предобученной модели MobileNetV2.
    """
    
    def __init__(self) -> None:
        """
        Инициализирует модель MobileNetV2 без верхних слоев классификации.
        
        Использует предварительно обученные на ImageNet веса и усреднение (average pooling)
        для получения одномерных векторов признаков.
        """
        # Фиксированный размер входного изображения для MobileNetV2
        self.input_size = (224, 224)
        
        # Создаем модель
        self.model = MobileNetV2(
            weights="imagenet",           # Используем предобученные веса
            include_top=False,            # Не включаем верхние слои классификации
            pooling="avg",                # Используем усреднение для получения одномерного вектора
            input_shape=(*self.input_size, 3)  # Размер входного изображения (224x224x3)
        )
        
    def get_embedding(self, image: np.ndarray) -> List[float]:
        """
        Выполняет эмбеддинг изображения, возвращая его векторное представление.
        
        Параметры:
            image (np.ndarray): Изображение в формате numpy array (RGB).
            
        Возвращает:
            List[float]: Векторное представление изображения в виде списка чисел с плавающей точкой.
        """
        # Изменяем размер изображения до требуемого для модели
        processed_img = cv2.resize(image, self.input_size)
        
        # Предобработка изображения для модели MobileNetV2
        processed_img = preprocess_input(processed_img)
        
        # Добавляем размерность батча (1 изображение)
        img_batch = np.expand_dims(processed_img, axis=0)
        
        # Измеряем время выполнения
        start_time = time.time()  # Начало замера времени
        
        # Получаем эмбеддинг с помощью модели (отключаем вывод прогресса)
        embedding = self.model.predict(img_batch, verbose=0)[0]
        
        end_time = time.time()  # Конец замера времени
        duration = end_time - start_time
        print(f"Время создания вектора: {duration:.6f} секунд")
        
        # Преобразуем numpy array в список и возвращаем
        return embedding.tolist()
        
    def get_embedding_with_time(self, image: np.ndarray) -> Tuple[List[float], float]:
        """
        Выполняет эмбеддинг изображения и возвращает векторное представление вместе с временем выполнения.
        
        Параметры:
            image (np.ndarray): Изображение в формате numpy array (RGB).
            
        Возвращает:
            Tuple[List[float], float]: Кортеж из векторного представления изображения и времени (в секундах),
                                      затраченного на создание вектора.
        """
        # Изменяем размер изображения до требуемого для модели
        processed_img = cv2.resize(image, self.input_size)
        
        # Предобработка изображения для модели MobileNetV2
        processed_img = preprocess_input(processed_img)
        
        # Добавляем размерность батча (1 изображение)
        img_batch = np.expand_dims(processed_img, axis=0)
        
        # Измеряем время выполнения
        start_time = time.time()  # Начало замера времени
        
        # Получаем эмбеддинг с помощью модели (отключаем вывод прогресса)
        embedding = self.model.predict(img_batch, verbose=0)[0]
        
        end_time = time.time()  # Конец замера времени
        duration = end_time - start_time
        
        # Преобразуем numpy array в список и возвращаем вместе с временем выполнения
        return embedding.tolist(), duration
    
model = Model()
