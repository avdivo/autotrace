"""
Модуль предоставляет набора инструментов для работы с изображениями интерфейса
- делает скриншот
- находит элементы интерфейса на изображении
"""
import cv2
import time
import numpy as np
from typing import List, Tuple, Optional
import mss

import settings
from hash_function import compute_dhash_vector, hex_to_dhash_vector, cosine_similarity


class ScreenCaptureError(Exception):
    """Класс для ошибок захвата экрана"""
    pass


class ScreenCapturer:
    """
    Класс для захвата скриншотов с экрана.
    """

    def __init__(self, monitor_number: int = 0):
        """
        Инициализация захвата экрана для указанного монитора.

        Args:
            monitor_number (int): Номер монитора (по умолчанию 0 - основной монитор)
        """
        self.monitor_number = monitor_number
        self.monitor = None
        self.sct = None
        self.hash_base_img = None

        self._initialize_capture()

    def _initialize_capture(self):
        """Инициализация средства захвата экрана и проверка монитора."""
        try:
            # Проверяем существование указанного монитора
            with mss.mss() as sct:
                if self.monitor_number >= len(sct.monitors):
                    raise ScreenCaptureError(
                        f"Монитор с номером {self.monitor_number} не существует. "
                        f"Доступны мониторы с номерами от 0 до {len(sct.monitors) - 1}"
                    )
                self.monitor = sct.monitors[self.monitor_number]

            # Инициализация средства захвата
            self.sct = mss.mss()

        except Exception as e:
            raise ScreenCaptureError(f"Ошибка инициализации захвата экрана: {str(e)}")

    def capture(self, region: dict = None) -> Optional[np.ndarray]:
        """
        Делает скриншот заданной области экрана и возвращает изображение в формате NumPy (BGR).

        Args:
            region (dict): Словарь с параметрами области захвата
                           (например, {"top": 0, "left": 0, "width": 1920, "height": 1080}).
                           Если None, захватывается весь монитор.

        Returns:
            Optional[np.ndarray]: Изображение экрана в формате NumPy (BGR) или None при ошибке.
        """
        if region is None:
            region = self.monitor

        try:
            # Делаем снимок экрана
            scr_img = self.sct.grab(region)

            # Преобразуем в NumPy массив
            img = np.asarray(scr_img)

            # Конвертируем RGB → BGR (если используется OpenCV)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            return img

        except Exception as e:
            print(f"Ошибка при создании скриншота: {e}")
            return None

    def __del__(self):
        """Закрытие соединения при удалении объекта"""
        if hasattr(self, 'sct') and self.sct:
            self.sct.close()


# Константы для адаптивной бинаризации
BLOCK_SIZE = 15  # Размер окна (должен быть нечетным)
C_MEAN = 3  # Смещение от среднего значения

# Настройки фильтра регионов
IOU_THRESHOLD = 0.5  # порог IoU для объединения пересекающихся прямоугольников
DIST_X_THRESHOLD = 4  # макс. горизонтальное расстояние между прямоугольниками
DIST_Y_THRESHOLD = 1  # макс. вертикальное расстояние между прямоугольниками
MIN_BOX_SIZE = 1  # минимальная ширина/высота прямоугольника

# Настройки алгоритма первичного поиска регионов
MIN_BOX_AREA = 37  # Минимальная площадь, при которой бокс не объединяется

# Порог схожести изображений
MIN_SIMILARITY_THRESHOLD = 0.72  # Минимальное совпадение (85%)


class UIRegions:
    """
    Управляет регионами пользовательского интерфейса, найденными на изображении.
    """
    class Region:
        """
        Представляет один регион (bounding box) и его dhash.

        Attributes:
            box (Tuple[int, int, int, int]): Координаты (x, y, w, h).
            dhash (str): Хэш изображения внутри региона.
        """

        def __init__(self, box: Tuple[int, int, int, int], image: np.ndarray):
            """
            Инициализирует регион и вычисляет dhash.

            Args:
                box (Tuple[int, int, int, int]): Прямоугольник (x, y, w, h).
                image (np.ndarray): Полное BGR-изображение, откуда берется подизображение.
            """
            self.box = box

            x, y, w, h = box
            self.dhash = compute_dhash_vector(image[y:y + h, x:x + w])

        def similarity_difference(self, other_hash: np.ndarray) -> float:
            """
            Вычисляет разницу косинусного сходства между текущим dhash и переданным хэшем.

            Args:
                other_hash (np.ndarray): Внешний вектор dhash (64-битный).

            Returns:
                float: Разница сходства (1 - косинусное расстояние) в диапазоне [0, 1].
            """
            res = cosine_similarity(self.dhash, other_hash)
            return res  # Чем ближе к 1, тем больше сходство

        def __str__(self):
            return f"box: {self.box} hash: {self.dhash_vector_to_hex(self.dhash)}"

    def __init__(self, image: np.ndarray):
        """
        Инициализирует список регионов по изображению.

        Args:
            image (np.ndarray): BGR-изображение для анализа.
        """
        self.original_image = image
        boxes = self.ui_detector()  # Поиск на изображении регионов содержащих элементы
        self.regions: List[UIRegions.Region] = [
            self.Region(box, image) for box in boxes
        ]

    def ui_detector(self) -> List[Tuple[int, int, int, int]]:
        """
        Поиск на изображении регионов содержащих элементы

        Returns:
            List[Tuple[int, int, int, int]]: Список прямоугольников (x, y, w, h), соответствующих найденным элементам.
        """

        def refine_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
            """
            Фильтрует и объединяет bounding boxes для лучшего выделения объектов.

            1. Удаляет вложенные боксы (оставляет только внешние).
            2. Объединяет близко расположенные боксы на основе IoU и расстояний.
            3. Исключает слишком маленькие боксы, чтобы убрать шум.

            Args:
                boxes (List[Tuple[int, int, int, int]]): Список bounding boxes (x, y, w, h).

            Returns:
                List[Tuple[int, int, int, int]]: Отфильтрованные и объединенные bounding boxes.
            """
            def is_inside(inner, outer):
                """ Проверяет, вложен ли один bounding box внутрь другого. """
                xi, yi, wi, hi = inner
                xo, yo, wo, ho = outer
                return xi > xo and yi > yo and xi + wi < xo + wo and yi + hi < yo + ho

            def iou(box1, box2):
                """ Вычисляет коэффициент пересечения областей (Intersection over Union, IoU). """
                x1, y1, w1, h1 = box1
                x2, y2, w2, h2 = box2

                # Вычисляем границы пересечения
                xi1, yi1 = max(x1, x2), max(y1, y2)
                xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                union_area = w1 * h1 + w2 * h2 - inter_area
                return inter_area / union_area if union_area > 0 else 0

            def are_close(box1, box2, dx_thresh, dy_thresh):
                """ Проверяет, находятся ли два bounding box рядом друг с другом. """
                x1, y1, w1, h1 = box1
                x2, y2, w2, h2 = box2

                # Горизонтальный и вертикальный зазор
                horiz_gap = max(x2 - (x1 + w1), x1 - (x2 + w2), 0)
                vert_gap = max(y2 - (y1 + h1), y1 - (y2 + h2), 0)
                return horiz_gap < dx_thresh and vert_gap < dy_thresh

            def merge_boxes(box1, box2):
                """ Объединяет два bounding box в один. """
                x1, y1, w1, h1 = box1
                x2, y2, w2, h2 = box2

                # Создаем новый объединенный bounding box
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1 + w1, x2 + w2)
                y_max = max(y1 + h1, y2 + h2)
                return (x_min, y_min, x_max - x_min, y_max - y_min)

            # Удаляем вложенные bounding boxes (оставляем только внешние)
            boxes = [box for i, box in enumerate(boxes)
                     if not any(is_inside(box, other) for j, other in enumerate(boxes) if i != j)]

            # Объединяем близкие bounding boxes
            merged = []
            while boxes:
                base = boxes.pop(0)
                changed = True
                while changed:
                    changed = False
                    i = 0
                    while i < len(boxes):
                        # Проверяем IoU и расстояние между боками
                        if iou(base, boxes[i]) > IOU_THRESHOLD or are_close(base, boxes[i], DIST_X_THRESHOLD,
                                                                            DIST_Y_THRESHOLD):
                            base = merge_boxes(base, boxes.pop(i))
                            changed = True
                            i = 0
                        else:
                            i += 1

                # Добавляем только достаточно большие боксы
                if base[2] >= MIN_BOX_SIZE and base[3] >= MIN_BOX_SIZE:
                    merged.append(base)

            return merged

        st = time.time()

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)  # Переводим изображение в оттенки серого
        gray = cv2.GaussianBlur(gray, (7, 7), 0)  # Чем больше размер ядра, тем сильнее сглаживание

        # Адаптивная бинаризация для всего изображения
        binary_image = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, C_MEAN
        )

        binary_image = 255 - binary_image  # Инверсия бинарного изображения

        # Контроль бинарного изображения
        # cv2.namedWindow("Binary image", cv2.WINDOW_FREERATIO)  # Окно можно изменять
        # cv2.imshow("Binary image", binary_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Поиск контуров
        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Перевод контуров в регионы
        bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

        # Удаление регионов с большой площадью
        filtered_boxes = [box for box in bounding_boxes if box[2] * box[3] < 5000]

        # Если ширина слишком большая относительно высоты (w/h > 10) или наоборот
        filtered_boxes = [box for box in filtered_boxes if 0.4 < (box[2] / box[3]) < 20]

        # Фильтрует и объединяет bounding boxes
        blocks = refine_boxes(filtered_boxes)

        # print("Время поиска регионов", time.time() - st)

        return blocks

    def process_click(self, click_x: int, click_y: int) -> Region:
        """
        Анализирует клик: ищет ближайший бокс, проверяет его площадь, объединяет соседние при необходимости.

        1. Находит ближайший прямоугольник.
        2. Проверяет его площадь:
            - Если `>= MIN_BOX_AREA`, просто возвращает его.
            - Если `< MIN_BOX_AREA`, объединяет его с соседними.
        3. Если соседей нет, возвращает найденный маленький бокс.

        Args:
            click_x (int): Координата X клика.
            click_y (int): Координата Y клика.
            boxes (List[Tuple[int, int, int, int]]): Список всех прямоугольников.

        Returns:
            Tuple[int, int, int, int]: Итоговый bounding box.
        """

        def find_nearest_box(click_x: int, click_y: int) -> UIRegions.Region:
            """
            Ищет ближайший bounding box к клику (по границе, а не по центру).

            1. Вычисляет расстояние от клика до ближайшей точки на границе каждого бокса.
            2. Выбирает бокс с минимальным расстоянием.

            Args:
                click_x (int): Координата X клика.
                click_y (int): Координата Y клика.

            Returns:
                Region: Найденный ближайший бокс.
            """
            if not self.regions:
                raise ValueError("Невозможно выполнить операцию: список регионов пуст.")

            def distance_to_box(cx: int, cy: int, x: int, y: int, w: int, h: int) -> float:
                # Находим ближайшую точку на границе прямоугольника
                nearest_x = np.clip(cx, x, x + w)
                nearest_y = np.clip(cy, y, y + h)
                return np.hypot(nearest_x - cx, nearest_y - cy)

            distances = [distance_to_box(click_x, click_y, *region.box) for region in self.regions]
            nearest_index = int(np.argmin(distances))
            return self.regions[nearest_index]

        nearest_region = find_nearest_box(click_x, click_y)
        x, y, w, h = nearest_region.box
        area = w * h

        if area < MIN_BOX_AREA:
            nearest_region = self.merge_nearby_boxes(nearest_region, merge_distance=20)
        return nearest_region

    def merge_nearby_boxes(self, target_box: Region, merge_distance: int) -> Region:
        """
        Объединяет соседние боксы, если их границы находятся в пределах merge_distance от центра target_box.

        Args:
            target_box (Tuple[int, int, int, int]): Исходный бокс.
            boxes (List[Tuple[int, int, int, int]]): Список всех боксов.
            merge_distance (int): Расстояние от центра данного блока до границ соседних для объединения

        Returns:
            Tuple[int, int, int, int]: Объединенный bounding box.
        """
        cx = target_box.box[0] + target_box.box[2] // 2
        cy = target_box.box[1] + target_box.box[3] // 2

        merged = []
        for region in self.regions:
            # Проверка: находится ли центр в пределах MERGE_DISTANCE от границы этого бокса
            x, y, w, h = region.box
            nearest_x = np.clip(cx, x, x + w)
            nearest_y = np.clip(cy, y, y + h)
            dist = np.hypot(cx - nearest_x, cy - nearest_y)

            if dist <= merge_distance:
                merged.append((x, y, w, h))

        if not merged:
            return target_box

        # Объединение в один бокс
        min_x = min(x for x, y, w, h in merged)
        min_y = min(y for x, y, w, h in merged)
        max_x = max(x + w for x, y, w, h in merged)
        max_y = max(y + h for x, y, w, h in merged)

        # Возвращаем новый регион
        return self.Region((min_x, min_y, max_x - min_x, max_y - min_y), self.original_image)

    def find_best_matching_regions(self, hex_hash: str) -> List[Region]:
        """
        Ищет регионы с максимальным совпадением по dhash, фильтруя по порогу MIN_SIMILARITY_THRESHOLD.

        Args:
            hex_hash (str): Хэш изображения в шестнадцатеричном формате (16 символов).

        Returns:
            List[Region]: Список регионов с максимальным совпадением, проходящих порог схожести.
        """
        binary_hash = hex_to_dhash_vector(hex_hash)  # Конвертируем hex → бинарный

        # Вычисляем схожесть для каждого региона
        similarity_scores = [(region, region.similarity_difference(binary_hash)) for region in self.regions]

        # Находим максимальное совпадение
        best_similarity = max(similarity_scores, key=lambda r: r[1])[1]
        # print("Лучшее сходство", best_similarity)
        if best_similarity < MIN_SIMILARITY_THRESHOLD:
            return []
        # Возвращаем только регионы, которые проходят минимальный порог
        # for i in similarity_scores:
        #     print("Схожесть найденного изображения ", i[1])
        return [region for region, score in similarity_scores if score == best_similarity]

    def find_by_extended_regions(self, regions: List[Region], hex_hash: str) -> Region or None:
        """
        Перебирает регионы, расширяем их и ищет лучшее совпадение с данным хэшем

        Args:
            hex_hash (str): Хэш изображения в шестнадцатеричном формате (16 символов)
            regions List[Region]: Регионы, среди которых ведется поиск, один из них возвращается.

        Returns:
            Region: Регион с максимальным совпадением, проходящий порог схожести, или первый, если их несколько
            None: Если ничего не найдено.
        """
        similarity_scores = []
        for region in regions:
            extended_region = self.merge_nearby_boxes(region, 30)  # Поиск расширенного региона (со стоящими рядом)
            binary_hash = hex_to_dhash_vector(hex_hash)  # Конвертируем hex → бинарный
            score = extended_region.similarity_difference(binary_hash)  # Результат сравнения расширенного региона с расширенным хэшем
            similarity_scores.append((region, score))  # Регион добавляем исходный, а результат от расширенного

        # Находим максимальное совпадение
        sort_similarity = sorted(similarity_scores, key=lambda r: r[1])
        # print("Лучшее сходство среди расширенных", sort_similarity[-1][1])
        if sort_similarity[-1][1] < MIN_SIMILARITY_THRESHOLD-0.2:
            return None  # Если не проходит порог схожести

        return sort_similarity[-1][0]




