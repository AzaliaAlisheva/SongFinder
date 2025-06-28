import numpy as np
import os
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorLoader:
    def __init__(self, qdrant_host: str = "qdrant", qdrant_port: int = 6333):
        """
        Инициализация загрузчика векторов
        
        Args:
            qdrant_host: Хост Qdrant
            qdrant_port: Порт Qdrant
        """
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.client = None
        
    def connect_to_qdrant(self, max_retries: int = 30, retry_delay: int = 2):
        """
        Подключение к Qdrant с повторными попытками
        
        Args:
            max_retries: Максимальное количество попыток подключения
            retry_delay: Задержка между попытками в секундах
        """
        for attempt in range(max_retries):
            try:
                self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
                # Проверяем подключение
                collections = self.client.get_collections()
                logger.info(f"Успешно подключился к Qdrant на {self.qdrant_host}:{self.qdrant_port}")
                return True
            except Exception as e:
                logger.warning(f"Попытка {attempt + 1}/{max_retries} подключения к Qdrant неудачна: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error("Не удалось подключиться к Qdrant после всех попыток")
                    return False
        return False
    
    def load_vectors_from_npy(self, file_path: str) -> np.ndarray:
        """
        Загрузка векторов из .npy файла
        
        Args:
            file_path: Путь к .npy файлу
            
        Returns:
            Массив векторов
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден")
        
        logger.info(f"Загружаю векторы из файла: {file_path}")
        vectors = np.load(file_path)
        
        if vectors.ndim != 2:
            raise ValueError(f"Ожидается 2D массив, получен {vectors.ndim}D массив")
        
        logger.info(f"Загружено {vectors.shape[0]} векторов размерности {vectors.shape[1]}")
        return vectors
    
    def create_collection(self, collection_name: str, vector_size: int):
        """
        Создание коллекции в Qdrant
        
        Args:
            collection_name: Название коллекции
            vector_size: Размерность векторов
        """
        try:
            # Проверяем, существует ли коллекция
            collections = self.client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if collection_name in existing_collections:
                logger.info(f"Коллекция '{collection_name}' уже существует")
                # Удаляем существующую коллекцию для перезаписи
                self.client.delete_collection(collection_name)
                logger.info(f"Удалена существующая коллекция '{collection_name}'")
            
            # Создаем новую коллекцию
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Создана коллекция '{collection_name}' с размерностью векторов {vector_size}")
            
        except Exception as e:
            logger.error(f"Ошибка при создании коллекции: {e}")
            raise
    
    def upload_vectors(self, collection_name: str, vectors: np.ndarray, batch_size: int = 100):
        """
        Загрузка векторов в Qdrant
        
        Args:
            collection_name: Название коллекции
            vectors: Массив векторов
            batch_size: Размер батча для загрузки
        """
        try:
            total_vectors = len(vectors)
            logger.info(f"Начинаю загрузку {total_vectors} векторов в коллекцию '{collection_name}'")
            
            # Загружаем векторы батчами
            for i in range(0, total_vectors, batch_size):
                batch_end = min(i + batch_size, total_vectors)
                batch_vectors = vectors[i:batch_end]
                
                # Создаем точки для загрузки
                points = [
                    PointStruct(
                        id=idx,
                        vector=vector.tolist(),
                        payload={"index": idx}
                    )
                    for idx, vector in enumerate(batch_vectors, start=i)
                ]
                
                # Загружаем батч
                self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
                logger.info(f"Загружено {batch_end}/{total_vectors} векторов")
            
            logger.info(f"Успешно загружены все {total_vectors} векторов в коллекцию '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке векторов: {e}")
            raise
    
    def run(self, npy_file_path: str, collection_name: str = "vectors"):
        """
        Основной метод для запуска загрузки
        
        Args:
            npy_file_path: Путь к .npy файлу
            collection_name: Название коллекции в Qdrant
        """
        try:
            # Подключаемся к Qdrant
            if not self.connect_to_qdrant():
                raise ConnectionError("Не удалось подключиться к Qdrant")
            
            # Загружаем векторы из файла
            vectors = self.load_vectors_from_npy(npy_file_path)
            
            # Создаем коллекцию
            self.create_collection(collection_name, vectors.shape[1])
            
            # Загружаем векторы
            self.upload_vectors(collection_name, vectors)
            
            logger.info("Загрузка векторов завершена успешно!")
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении: {e}")
            raise

def main():
    """Основная функция"""
    # Получаем параметры из переменных окружения
    npy_file = os.getenv("NPY_FILE_PATH", "/data/vectors.npy")
    collection_name = os.getenv("COLLECTION_NAME", "vectors")
    qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    
    logger.info(f"Параметры:")
    logger.info(f"  NPY файл: {npy_file}")
    logger.info(f"  Коллекция: {collection_name}")
    logger.info(f"  Qdrant хост: {qdrant_host}:{qdrant_port}")
    
    # Создаем и запускаем загрузчик
    loader = VectorLoader(qdrant_host=qdrant_host, qdrant_port=qdrant_port)
    loader.run(npy_file, collection_name)

if __name__ == "__main__":
    main() 