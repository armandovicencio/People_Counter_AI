import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import easyocr


class ObjectTracker:
    """
    Clase principal para el seguimiento de objetos y la aplicación de lógicas de análisis.
    Utiliza un modelo YOLOv8 para la detección y seguimiento.
    """

    def __init__(self, model_path='yolov8n.pt'):
        """
        Inicializa el tracker.

        Args:
            model_path (str): Ruta al archivo del modelo YOLOv8 (.pt).
        """
        try:
            # Carga el modelo YOLOv8 desde la ruta especificada.
            self.model = YOLO(model_path)
            print(f"✅ Modelo '{model_path}' cargado exitosamente.")
        except Exception as e:
            print(f"❌ Error al cargar el modelo YOLO: {e}")
            # Lanza la excepción para que la aplicación principal pueda manejarla.
            raise

        # --- Variables de Estado del Tracker ---
        # defaultdict(list) crea una lista vacía para cualquier ID de track nuevo.
        # Almacena el historial de posiciones (centro del bounding box) de cada objeto seguido.
        self.track_history = defaultdict(list)
        self.reader = None  # Inicializar como None por defecto

        # --- Inicialización del Lector OCR ---
        # 'en' es para inglés. Se pueden añadir otros idiomas como ['en', 'es'].
        # gpu=False fuerza el uso de CPU si no se dispone de una GPU compatible.
        try:
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("✅ Lector OCR (EasyOCR) inicializado en CPU.")
        except Exception as e:
            print(f"⚠️ Advertencia: No se pudo inicializar EasyOCR. La funcionalidad de OCR estará deshabilitada.")
            print(f"   Error de OCR: {e}")

    def process_frame(self, frame, tracking_enabled=True):
        """
        Procesa un único frame de video para obtener detecciones o tracks.

        Args:
            frame (np.ndarray): El frame de video de entrada (en formato BGR).
            tracking_enabled (bool): Si es True, usa el seguimiento de objetos. Si es False, solo detecta.

        Returns:
            Un objeto de resultados de Ultralytics (results[0]).
        """
        if tracking_enabled:
            # model.track() es la función clave de Ultralytics para el seguimiento.
            # `persist=True` le indica al tracker que este frame es consecutivo al anterior.
            results = self.model.track(frame, persist=True, verbose=False)
        else:
            # `model()` realiza solo la detección en el frame actual.
            results = self.model(frame, verbose=False)

        # El resultado es una lista, usualmente con un solo elemento.
        return results[0]

    def _get_tracks_and_boxes(self, results):
        """
        Método auxiliar para extraer IDs de seguimiento y bounding boxes de los resultados.
        """
        # Extrae las coordenadas de las cajas delimitadoras en formato (x_centro, y_centro, ancho, alto).
        boxes = results.boxes.xywh.cpu()

        # Verifica si hay IDs de seguimiento en el frame actual.
        if results.boxes.id is None:
            return [], []  # Retorna listas vacías si no hay objetos seguidos.

        # Convierte los IDs de seguimiento a una lista de enteros.
        track_ids = results.boxes.id.int().cpu().tolist()
        return boxes, track_ids

    def count_objects_in_region(self, results, region_points):
        """
        Cuenta los objetos cuyo centro del bounding box está dentro de un polígono definido.

        Args:
            results: El objeto de resultados de Ultralytics para el frame actual.
            region_points (np.array): Un array de NumPy con los vértices del polígono.

        Returns:
            Una lista con los track_ids de los objetos que están dentro de la región.
        """
        in_region_ids = []
        boxes, track_ids = self._get_tracks_and_boxes(results)

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            center_point = (float(x), float(y))

            # `cv2.pointPolygonTest` es una función eficiente para esta comprobación.
            # Devuelve +1 si está dentro, -1 si está fuera, 0 si está en el borde.
            is_inside = cv2.pointPolygonTest(region_points, center_point, False) >= 0
            if is_inside:
                in_region_ids.append(track_id)

        return in_region_ids

    def count_line_crossings(self, results, line_points):
        """
        Cuenta los objetos que cruzan una línea definida, usando el historial de seguimiento.

        Args:
            results: El objeto de resultados de Ultralytics para el frame actual.
            line_points (tuple): Una tupla con dos puntos ((x1, y1), (x2, y2)) que definen la línea.

        Returns:
            Una tupla (count_in, count_out) con el número de cruces en cada dirección.
        """
        boxes, track_ids = self._get_tracks_and_boxes(results)
        count_in = 0
        count_out = 0

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            center_point = (float(x), float(y))

            # Actualiza el historial de posiciones para el objeto actual.
            track = self.track_history[track_id]
            track.append(center_point)
            # Mantenemos un historial limitado para evitar consumo excesivo de memoria.
            if len(track) > 30:
                track.pop(0)

            # Para detectar un cruce, necesitamos al menos dos puntos en el historial.
            if len(track) >= 2:
                prev_point = track[-2]
                current_point = track[-1]

                # Comprueba si el segmento de línea del movimiento del objeto intersecta la línea de conteo.
                if self._check_intersection(prev_point, current_point, line_points):
                    # Heurística simple para determinar la dirección del cruce.
                    # Comparamos la posición X del objeto con la X promedio de la línea.
                    # Esto funciona bien para líneas mayormente verticales.
                    # Para una mayor precisión, se podría usar el producto cruzado de vectores.
                    line_x_avg = (line_points[0][0] + line_points[1][0]) / 2
                    if prev_point[0] < line_x_avg and current_point[0] >= line_x_avg:
                        count_in += 1
                    elif prev_point[0] > line_x_avg and current_point[0] <= line_x_avg:
                        count_out += 1

        return count_in, count_out

    def _check_intersection(self, p1, p2, line_points):
        """
        Comprueba si el segmento de línea (p1, p2) se cruza con el segmento `line_points`.
        Algoritmo estándar de intersección de segmentos de línea.
        """
        p3, p4 = line_points

        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0: return 0  # Colineal
            return 1 if val > 0 else 2  # Sentido horario o antihorario

        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)

        # Caso general: si las orientaciones cambian, las líneas se cruzan.
        if o1 != o2 and o3 != o4:
            return True

        # Casos especiales para puntos colineales (no implementados aquí por simplicidad).
        return False

    def extract_text_from_roi(self, frame, ocr_roi):
        """
        (Para Fase 5) Extrae texto de una Región de Interés (ROI) en el frame.

        Args:
            frame (np.ndarray): El frame de video de entrada.
            ocr_roi (list): Una lista [x1, y1, x2, y2] que define la región para OCR.

        Returns:
            Una cadena de texto con el texto extraído concatenado.
        """
        if self.reader is None:
            return "OCR NO DISPONIBLE"

        x1, y1, x2, y2 = ocr_roi
        # Recorta la región de interés del frame.
        roi_img = frame[y1:y2, x1:x2]

        # --- Pre-procesamiento de la Imagen para Mejorar el OCR ---
        # Convertir a escala de grises suele ayudar.
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        # Aplicar un umbral (thresholding) puede limpiar la imagen.
        # Los valores 150 y 255 pueden necesitar ajustes según el video.
        _, thresh_roi = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY)

        # Usa el lector de EasyOCR para encontrar texto en la ROI procesada.
        results = self.reader.readtext(thresh_roi)

        # Concatena todos los fragmentos de texto encontrados en una sola cadena.
        return " ".join([res[1] for res in results])

