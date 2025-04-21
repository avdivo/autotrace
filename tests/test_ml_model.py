#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Тесты для модуля ml_model
=========================

Модуль содержит тесты для проверки функциональности класса Model из модуля ml_model.
Тесты проверяют корректность создания модели и генерации эмбеддингов изображений.
"""

import os
import pytest
import numpy as np
import cv2
import time  # Импортируем модуль time
from ml_model import Model
from exceptions import ModelError, EmbeddingError

class TestModel:
    """Тесты для класса Model."""

    @pytest.fixture(scope="module")
    def model(self):
        """Фикстура для создания экземпляра класса Model с кэшированием."""
        return Model()

    def test_model_initialization(self, model):
        """Проверка корректности инициализации модели."""
        start_time = time.time()  # Начало замера времени
        # Проверяем, что модель создана
        assert model is not None
        
        # Проверяем, что атрибуты модели корректно инициализированы
        assert model.input_size == (224, 224)
        assert model.model is not None
        end_time = time.time()  # Конец замера времени
        print(f"Время инициализации модели: {end_time - start_time:.6f} секунд")

    def test_get_embedding(self, model, sample_image):
        """Проверка корректности работы метода get_embedding."""
        start_time = time.time()  # Начало замера времени
        
        # Получаем эмбеддинг для тестового изображения
        embedding = model.get_embedding(sample_image)
        
        end_time = time.time()  # Конец замера времени
        print(f"Время создания вектора: {end_time - start_time:.6f} секунд")
        
        # Проверяем, что эмбеддинг является списком
        assert isinstance(embedding, list)
        
        # Проверяем, что эмбеддинг содержит числа с плавающей точкой
        assert all(isinstance(x, float) for x in embedding)
        
        # Проверяем размерность эмбеддинга (для MobileNetV2 это 1280)
        assert len(embedding) == 1280

    def test_embedding_consistency(self, model, different_images):
        """Проверка согласованности эмбеддингов."""
        img1, _ = different_images
        
        start_time = time.time()  # Начало замера времени
        emb1 = model.get_embedding(img1)
        emb2 = model.get_embedding(img1)  # Тот же самый образец
        end_time = time.time()  # Конец замера времени
        print(f"Время получения эмбеддингов: {end_time - start_time:.6f} секунд")
        
        # Проверяем, что эмбеддинги идентичны
        assert np.allclose(emb1, emb2)

    def test_embedding_difference(self, model, different_images):
        """Проверка различия эмбеддингов для разных изображений."""
        img1, img2 = different_images
        
        start_time = time.time()  # Начало замера времени
        emb1 = model.get_embedding(img1)
        emb2 = model.get_embedding(img2)
        end_time = time.time()  # Конец замера времени
        print(f"Время получения эмбеддингов для разных изображений: {end_time - start_time:.6f} секунд")
        
        # Проверяем, что эмбеддинги различаются
        cosine_distance = 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        assert cosine_distance > 0.1

    def test_embedding_format_compatibility_with_db(self, model, sample_image):
        """Проверка совместимости формата эмбеддинга с форматом БД."""
        start_time = time.time()  # Начало замера времени
        embedding = model.get_embedding(sample_image)
        end_time = time.time()  # Конец замера времени
        print(f"Время получения эмбеддинга для проверки формата: {end_time - start_time:.6f} секунд")
        
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)
        
        screen_record = {
            "ids": "test_screen_id",
            "embeddings": [embedding]
        }
        
        sample_record = {
            "ids": "test_sample_id",
            "embeddings": [embedding],
            "metadatas": {"screen_id": "test_screen_id"}
        }
        
        assert isinstance(screen_record["embeddings"][0], list)
        assert isinstance(sample_record["embeddings"][0], list)

    def test_invalid_image(self, model):
        """Проверка обработки некорректного изображения."""
        invalid_img = np.array([])
        
        with pytest.raises(Exception):
            model.get_embedding(invalid_img)

if __name__ == "__main__":
    pytest.main(["-v"])