import cv2
import numpy as np
import time
import multiprocessing
from multiprocessing import shared_memory
import logging
import settings

from exceptions import ScreenCaptureError, MemoryError, MonitorError
from ui_detector import ScreenCapturer
from hash_function import compute_dhash_vector, cosine_similarity


class ScreenMonitor:
    """
    Класс для мониторинга экрана и сохранения скриншотов в shared_memory.
    Создает отдельный процесс, который делает скриншоты через заданные промежутки времени.
    Если новый скриншот существенно отличается от предыдущего, он сохраняется в shared_memory.
    """
    def __init__(self, monitor_number=settings.SCREEN_MONITOR_NUMBER,
                 width=settings.SCREEN_RESOLUTION[0],
                 height=settings.SCREEN_RESOLUTION[1]):
        """
        Инициализация монитора экрана.
        
        Args:
            monitor_number (int): Номер монитора для захвата
            width (int): Ширина буфера изображения
            height (int): Высота буфера изображения
            
        Raises:
            ScreenCaptureError: Если указанный монитор не существует
            MemoryError: Если возникла ошибка при создании shared memory
        """
        self.monitor_number = monitor_number
        self.width = width
        self.height = height

    @staticmethod
    def _monitor_screen_process(interval, monitor_number, diff_threshold, 
                               shared_memory_name, buffer_shape, buffer_size):
        """
        Процесс для мониторинга экрана.
        
        Args:
            interval (float): Интервал между скриншотами
            monitor_number (int): Номер монитора для захвата
            diff_threshold (int): Порог различия для определения изменения экрана
            shared_memory_name (str): Имя области shared memory
            buffer_shape (tuple): Форма буфера (высота, ширина, каналы)
            buffer_size (int): Размер буфера в байтах
        """
        # Подготовка средства захвата экрана
        capturer = ScreenCapturer(monitor_number=monitor_number)

        hash_base_img = None  # Хеш предыдущего изображения
        
        # Подключаемся к уже созданной в основном процессе shared memory
        try:
            # Shared memory должна быть уже создана в основном процессе
            shm = shared_memory.SharedMemory(name=shared_memory_name)
            logger.info(f"Процесс мониторинга подключился к shared memory '{shared_memory_name}'")
        except FileNotFoundError:
            # Если по каким-то причинам shared memory не создана, создаем её
            logger.warning(f"Shared memory '{shared_memory_name}' не найдена, создаем новую")
            try:
                shm = shared_memory.SharedMemory(
                    name=shared_memory_name, 
                    create=True, 
                    size=buffer_size
                )
                # Заполняем нулями - изначально память пуста
                shm.buf[:] = b'\0' * buffer_size
            except Exception as e:
                logger.error(f"Ошибка при создании shared memory: {e}")
                raise MemoryError(f"Ошибка создания shared memory: {str(e)}")
        
        # Создаем numpy массив, который будет ссылаться на shared memory
        # Важно: хотя массив многомерный (3D), в памяти он хранится линейно
        shm_array = np.ndarray(shape=buffer_shape, dtype=np.uint8, buffer=shm.buf)
        
        logger.info(f"Процесс мониторинга экрана запущен.")
        
        try:
            while True:
                start_time = time.time()  # Засекаем время для расчета интервала

                # Захват всего экрана
                img = capturer.capture()
                
                # Изменяем размер, если скриншот не соответствует размеру буфера
                if img.shape[0] != buffer_shape[0] or img.shape[1] != buffer_shape[1]:
                    img = cv2.resize(img, (buffer_shape[1], buffer_shape[0]))

                # Проверяем, что количество каналов совпадает
                if img.shape[2] != buffer_shape[2]:
                    # Преобразуем к нужному формату (BGRA)
                    if img.shape[2] == 3:  # BGR -> BGRA
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    elif img.shape[2] > 4:  # Берем только первые 4 канала
                        img = img[:, :, :4]
                
                # Получаем хеш изображения для сравнения
                hash_img = compute_dhash_vector(img)

                # Первый скриншот или значительное изменение экрана
                if hash_base_img is None or cosine_similarity(hash_base_img[0], hash_img[0]) > diff_threshold:
                    time.sleep(0.2)  # Задержка для стабилизации изображения
                    # Делаем скриншот экрана повторно
                    hash_img = capturer.capture()

                    logger.info("Обнаружено изменение экрана или первый скриншот")
                    hash_base_img = hash_img  # Сохраняем новый хеш
                    
                    # Копируем данные изображения в shared memory через numpy массив
                    # Это эффективнее, чем копировать по байтам
                    shm_array[:] = img[:]
                    logger.info(f"Скриншот сохранен в shared memory. Размер: {img.shape}")
                
                # Вычисляем, сколько нужно спать до следующего скриншота
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                logger.info(f"Время выполнения скриншота: {elapsed} секунд")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Процесс мониторинга экрана остановлен пользователем")
        except Exception as e:
            logger.exception(f"Ошибка в процессе мониторинга экрана: {e}")
            raise
            raise MonitorError(f"Ошибка в процессе мониторинга: {str(e)}")
        finally:
            # Освобождаем ресурсы
            if shm is not None:
                try:
                    shm.close()  # Закрываем только доступ к shared memory, не удаляем её
                except:
                    pass

    def get_screenshot(self):
        """
        Получает последний скриншот из shared memory.
        
        Returns:
            numpy.ndarray or None: Массив изображения или None, если скриншота нет
            
        Raises:
            MemoryError: Если shared memory не инициализирована или возникла ошибка доступа
        """
        try:
            # Проверяем, что shared memory активна
            if self.shm is None:
                logger.warning("Shared memory не инициализирована или была закрыта")
                return None
            
            # Создаем numpy массив, который ссылается на данные в shared memory
            # Преимущество numpy: мы можем представить линейные данные как многомерный массив
            img = np.ndarray(shape=self.buffer_shape, dtype=np.uint8, buffer=self.shm.buf)
            
            # Проверяем, не пустой ли массив (все значения равны 0)
            if np.all(img == 0):
                logger.debug("Данные в shared memory пусты")
                return None
                
            # Создаем копию данных, чтобы избежать проблем с доступом к shared memory
            # после её очистки или закрытия
            img_copy = img.copy()
            
            # Очищаем shared memory после считывания данных
            self.clear_shared_memory()
            
            return img_copy
        except Exception as e:
            logger.error(f"Ошибка при получении скриншота: {e}")
            # Не выбрасываем исключение, просто возвращаем None, чтобы не прерывать работу программы
            return None
    
    def clear_shared_memory(self):
        """Очищает данные в shared memory, не удаляя саму область памяти"""
        if self.shm is not None:
            try:
                # Заполняем буфер нулями
                self.shm.buf[:] = b'\0' * self.buffer_size
                logger.info("Данные в shared memory очищены")
            except Exception as e:
                logger.error(f"Ошибка при очистке shared memory: {e}")
                raise MemoryError(f"Ошибка при очистке shared memory: {str(e)}")


# Пример использования:
if __name__ == "__main__":
    # Создаем и запускаем монитор экрана
    # Можно указать нестандартный размер: monitor = ScreenMonitor(width=1280, height=720)
    monitor = ScreenMonitor(interval=0.5)
    monitor.start()
    
    try:
        # Основной цикл программы - получение и отображение скриншотов
        while True:
            img = monitor.get_screenshot()
            if img is not None:
                # Отображаем скриншот
                cv2.imshow("Screen Monitor", img)
                # clear_shared_memory уже вызывается внутри get_screenshot()
            
            # Выход при нажатии 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.1)  # Небольшая пауза чтобы не нагружать CPU
            
    except KeyboardInterrupt:
        print("Программа остановлена пользователем")
    finally:
        # Убедимся, что ресурсы корректно освобождены
        cv2.destroyAllWindows()
        monitor.stop() 