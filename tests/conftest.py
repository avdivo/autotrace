#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Общие фикстуры pytest для тестирования модулей проекта.
Содержит многократно используемые объекты для тестов.
"""

import pytest
import numpy as np

@pytest.fixture
def sample_image():
    """Общая фикстура для создания тестового изображения."""
    return np.zeros((64, 64, 3), dtype=np.uint8)

@pytest.fixture
def different_images():
    """Фикстура для создания пары разных изображений."""
    img1 = np.ones((64, 64, 3), dtype=np.uint8) * 255  # Белое изображение
    img2 = np.zeros((64, 64, 3), dtype=np.uint8)      # Черное изображение
    return img1, img2 