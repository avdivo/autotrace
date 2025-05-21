#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для работы с ChromaDB (chroma_db.py)
==========================================

Модуль предоставляет класс Chroma, который обеспечивает взаимодействие с базой данных ChromaDB.
База данных используется для хранения и поиска векторных представлений изображений экрана
и их сегментов.

При импорте модуля создается экземпляр класса Chroma, который может использоваться
в различных частях программы для работы с базой данных.
"""

import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Union, Any
import settings
from exceptions import ChromaDBError, CollectionNotFoundError, DocumentNotFoundError, DuplicateIDError

class Chroma:
    """
    Класс для работы с векторной базой данных ChromaDB.
    
    Обеспечивает функциональность создания, поиска и управления записями
    экранов и образцов в базе данных. Используется для сравнения изображений
    и определения, находится ли система на известном экране.
    """
    
    def __init__(self, 
                 persist_directory: str = settings.CHROMA_PERSIST_DIRECTORY,
                 port: int = settings.CHROMA_PORT,
                 screen_collection_name: str = settings.CHROMA_SCREEN_COLLECTION,
                 sample_collection_name: str = settings.CHROMA_SAMPLE_COLLECTION) -> None:
        """
        Инициализирует соединение с базой данных ChromaDB.
        
        Параметры:
            persist_directory (str): Директория для хранения данных ChromaDB.
            port (int): Порт для запуска сервера ChromaDB.
            screen_collection_name (str): Название коллекции для экранов.
            sample_collection_name (str): Название коллекции для образцов.
        """
        self.persist_directory = persist_directory
        self.port = port
        self.screen_collection_name = screen_collection_name
        self.sample_collection_name = sample_collection_name
        self.client = None
        self.screen_collection = None
        self.sample_collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Инициализирует клиент ChromaDB."""
        try:
            # Создаем директорию для хранения данных ChromaDB, если она не существует
            os.makedirs(self.persist_directory, exist_ok=True)

            # Инициализируем клиент ChromaDB, который запускает сервер на указанном порту
            self.client = chromadb.Client(Settings(
                chroma_server_host="localhost",
                chroma_server_http_port=self.port,
                persist_directory=self.persist_directory,
                is_persistent=True
            ))

            # Получаем или создаем коллекции для экранов и образцов
            self.screen_collection = self.client.get_or_create_collection(
                name=self.screen_collection_name,
                metadata={"hnsw:space": "cosine"}  # Используем косинусное расстояние для сравнения векторов
            )

            self.sample_collection = self.client.get_or_create_collection(
                name=self.sample_collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        except Exception as e:
            raise ChromaDBError(f"Ошибка при инициализации ChromaDB: {str(e)}")
    
    def shutdown(self):
        """Останавливает сервер ChromaDB и освобождает ресурсы."""
        if self.client is not None:
            try:
                if hasattr(self.client, 'persist'):
                    self.client.persist()
                
                if hasattr(self.client, '_system'):
                    self.client._system.stop()
                elif hasattr(self.client, 'reset_state'):
                    self.client.reset_state()
            except Exception as e:
                print(f"Предупреждение: Не удалось корректно остановить клиент ChromaDB: {str(e)}")
            finally:
                self.client = None
                self.screen_collection = None
                self.sample_collection = None
    
    def get_screen_id(self, embedding: List[float]) -> Optional[str]:
        """
        Ищет экран по вектору в базе данных.
        
        Проверяет, есть ли в базе данных vector, близкий к переданному.
        Степень близости определяется параметром SCREEN_SIMILARITY_THRESHOLD из модуля settings.
        
        Параметры:
            embedding (List[float]): Векторное представление экрана для поиска.
            
        Возвращает:
            Optional[str]: Идентификатор экрана или None, если подходящий экран не найден.
        """
        try:
            results = self.screen_collection.query(
                query_embeddings=[embedding],
                n_results=1,
                include=["distances", "metadatas"]
            )
            
            if results["distances"] and results["distances"][0]:
                distance = results["distances"][0][0]
                if distance < (1 - settings.SCREEN_SIMILARITY_THRESHOLD):
                    return results["ids"][0][0]
            
            return None
        
        except Exception as e:
            raise ChromaDBError(f"Ошибка при поиске screen_id: {str(e)}")
    
    def create_screen(self, screen_id: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Создает новую запись экрана в базе данных.
        
        Параметры:
            screen_id (str): Уникальный идентификатор экрана.
            embedding (List[float]): Векторное представление экрана.
            metadata (Optional[Dict[str, Any]]): Метаданные экрана (опционально).
        """
        try:
            # Проверяем, не существует ли уже экран с таким ID
            try:
                existing_screen = self.screen_collection.get(ids=screen_id)
                if existing_screen and len(existing_screen['ids']) > 0:
                    raise DuplicateIDError(f"Экран с id {screen_id} уже существует")
            except Exception as check_e:
                # Если ошибка не связана с проверкой существования, игнорируем её
                if not "not found" in str(check_e).lower():
                    raise check_e
            
            # Добавляем запись в коллекцию экранов
            if metadata:
                self.screen_collection.add(
                    ids=screen_id,
                    embeddings=[embedding],
                    metadatas=metadata
                )
            else:
                self.screen_collection.add(
                    ids=screen_id,
                    embeddings=[embedding]
                )
        
        except DuplicateIDError as e:
            raise e
        except Exception as e:
            error_message = str(e).lower()
            if "already exists" in error_message or "duplicate" in error_message or "unique" in error_message:
                raise DuplicateIDError(f"Экран с id {screen_id} уже существует")
            raise ChromaDBError(f"Ошибка при создании экрана: {str(e)}")
    
    # def get_sample_id(self, embedding: List[float], metadata: Dict[str, str]) -> Optional[str]:
    #     """
    #     Ищет образец по вектору и метаданным в базе данных.
    #
    #     Проверяет, есть ли в базе данных vector, близкий к переданному,
    #     с указанным screen_id в метаданных.
    #
    #     Параметры:
    #         embedding (List[float]): Векторное представление образца для поиска.
    #         metadata (Dict[str, str]): Метаданные, включающие screen_id.
    #
    #     Возвращает:
    #         Optional[str]: Идентификатор образца или None, если подходящий образец не найден.
    #     """
    #     try:
    #         # Проверяем, что в метаданных есть screen_id
    #         if "screen_id" not in metadata:
    #             raise ValueError("Метаданные должны содержать ключ 'screen_id'")
    #
    #         # Выполняем запрос к коллекции образцов с фильтрацией по screen_id
    #         results = self.sample_collection.query(
    #             query_embeddings=[embedding],
    #             n_results=1,
    #             where={"screen_id": metadata["screen_id"]},
    #             include=["distances", "metadatas"]
    #         )
    #
    #         # Проверяем, есть ли результаты и достаточно ли они близки
    #         if results["distances"] and results["distances"][0]:
    #             distance = results["distances"][0][0]
    #             if distance < (1 - settings.SAMPLE_SIMILARITY_THRESHOLD):  # Преобразование сходства в расстояние
    #                 return results["ids"][0][0]
    #
    #         return None
    #
    #     except ValueError as e:
    #         raise e
    #     except Exception as e:
    #         raise ChromaDBError(f"Ошибка при поиске sample_id: {str(e)}")
    
    def create_sample(self, sample_id: str, metadata: Dict[str, str]) -> None:
        """
        Создает новую запись образца в базе данных.
        
        Параметры:
            sample_id (str): Уникальный идентификатор образца.
            metadata (Dict[str, str]): Метаданные образца, включающие screen_id.
        """
        try:
            if "screen_id" not in metadata:
                raise ValueError("Метаданные должны содержать ключ 'screen_id'")
            
            try:
                existing_sample = self.sample_collection.get(ids=sample_id)
                if existing_sample and len(existing_sample['ids']) > 0:
                    raise DuplicateIDError(f"Образец с id {sample_id} уже существует")
            except Exception as check_e:
                if not "not found" in str(check_e).lower():
                    raise check_e

            self.sample_collection.add(
                ids=sample_id,
                embeddings=[[0.0] * 64],
                metadatas=metadata
            )
        
        except ValueError as e:
            raise e
        except DuplicateIDError as e:
            raise e
        except Exception as e:
            error_message = str(e).lower()
            if "already exists" in error_message or "duplicate" in error_message or "unique" in error_message:
                raise DuplicateIDError(f"Образец с id {sample_id} уже существует")
            raise ChromaDBError(f"Ошибка при создании образца: {str(e)}")
    
    def get_sample(self, sample_id: str, metadata: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """
        Получает информацию об образце по его идентификатору и метаданным.
        
        Параметры:
            sample_id (str): Идентификатор образца.
            metadata (Optional[Dict[str, str]]): Метаданные для фильтрации.
            
        Возвращает:
            Optional[Dict[str, Any]]: Словарь с информацией об образце или None, если образец не найден.
        """
        try:
            where_filter = metadata if metadata else None
            
            results = self.sample_collection.get(
                ids=sample_id,
                where=where_filter,
                include=["metadatas"]
            )

            if results["ids"]:
                return {
                    "id": results["ids"][0],
                    "metadata": results["metadatas"][0] if "metadatas" in results else None
                }
            
            return None
        
        except Exception as e:
            if "not found" in str(e).lower():
                return None
            raise ChromaDBError(f"Ошибка при получении образца: {str(e)}")


# Создаем экземпляр класса Chroma для использования в других модулях
chroma_db = Chroma() 
