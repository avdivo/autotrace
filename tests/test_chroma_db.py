#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Тесты для модуля chroma_db
=========================

Модуль содержит тесты для проверки функциональности класса Chroma из модуля chroma_db.
Тесты проверяют корректность создания, поиска и обновления записей в базе данных.
"""

import os
import pytest
import numpy as np
from chroma_db import Chroma
from exceptions import ChromaDBError, DuplicateIDError
import settings
import uuid

class TestChroma:
    """Тесты для класса Chroma."""

    @pytest.fixture(scope="module")
    def chroma(self):
        """Фикстура для создания экземпляра класса Chroma."""
        # Используем временную директорию для тестов
        test_dir = f"test_chroma_db_{uuid.uuid4().hex}"
        chroma_instance = Chroma(persist_directory=test_dir, port=8001)
        yield chroma_instance
        
        # Очистка после тестов
        chroma_instance.shutdown()  # Останавливаем сервер ChromaDB перед удалением директории
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)

    @pytest.fixture
    def sample_embedding(self):
        """Фикстура для создания тестового вектора."""
        return np.random.rand(128).tolist()

    @pytest.fixture
    def sample_screen_id(self):
        """Фикстура для создания идентификатора экрана."""
        return f"screen_{uuid.uuid4().hex}"
    
    @pytest.fixture
    def sample_metadata(self, sample_screen_id):
        """Фикстура для создания метаданных образца."""
        return {"screen_id": sample_screen_id}

    def test_initialization(self, chroma):
        """Проверка корректности инициализации ChromaDB."""
        assert chroma is not None
        assert chroma.client is not None
        assert chroma.screen_collection is not None
        assert chroma.sample_collection is not None

    def test_create_and_get_screen(self, chroma, sample_embedding, sample_screen_id):
        """Проверка создания и получения записи экрана."""
        # Создаем запись экрана
        chroma.create_screen(sample_screen_id, sample_embedding)
        
        # Ищем экран по тому же вектору (должен найтись)
        found_id = chroma.get_screen_id(sample_embedding)
        assert found_id == sample_screen_id
        
        # Создаем другой вектор (должен быть достаточно разным)
        different_embedding = np.random.rand(128).tolist()
        found_id = chroma.get_screen_id(different_embedding)
        assert found_id is None

    def test_create_duplicate_screen(self, chroma, sample_embedding, sample_screen_id):
        """Проверка ошибки при создании дубликата экрана."""
        # Создаем запись экрана
        chroma.create_screen(sample_screen_id, sample_embedding)
        
        # Пытаемся создать экран с тем же ID (должна быть ошибка)
        with pytest.raises(DuplicateIDError):
            chroma.create_screen(sample_screen_id, sample_embedding)

    def test_create_and_get_sample(self, chroma, sample_embedding, sample_screen_id, sample_metadata):
        """Проверка создания и получения записи образца."""
        # Создаем запись образца
        sample_id = f"sample_{uuid.uuid4().hex}"
        chroma.create_sample(sample_id, sample_embedding, sample_metadata)
        
        # Получаем образец по ID
        sample = chroma.get_sample(sample_id)
        assert sample is not None
        assert sample["id"] == sample_id
        assert len(sample["embedding"]) == len(sample_embedding)
        assert sample["metadata"]["screen_id"] == sample_metadata["screen_id"]
        
        # Ищем образец по вектору и метаданным
        found_id = chroma.get_sample_id(sample_embedding, sample_metadata)
        assert found_id == sample_id
        
        # Ищем по другому вектору (не должен найтись)
        different_embedding = np.random.rand(128).tolist()
        found_id = chroma.get_sample_id(different_embedding, sample_metadata)
        assert found_id is None

    def test_get_nonexistent_sample(self, chroma):
        """Проверка получения несуществующего образца."""
        # Пытаемся получить несуществующий образец
        sample = chroma.get_sample("nonexistent_sample_id")
        assert sample is None

    def test_create_sample_without_screen_id(self, chroma, sample_embedding):
        """Проверка ошибки при создании образца без screen_id в метаданных."""
        # Пытаемся создать образец без screen_id в метаданных
        sample_id = f"sample_{uuid.uuid4().hex}"
        with pytest.raises(ValueError):
            chroma.create_sample(sample_id, sample_embedding, {})

    def test_get_sample_id_without_screen_id(self, chroma, sample_embedding):
        """Проверка ошибки при поиске образца без screen_id в метаданных."""
        # Пытаемся найти образец без screen_id в метаданных
        with pytest.raises(ValueError):
            chroma.get_sample_id(sample_embedding, {})

if __name__ == "__main__":
    pytest.main(["-v"]) 