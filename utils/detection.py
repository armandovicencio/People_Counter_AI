import cv2
import numpy as np
import time
import os
import urllib.request
from collections import defaultdict

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch no está disponible")

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError as e:
    ULTRALYTICS_AVAILABLE = False
    print(f"❌ Ultralytics no está disponible: {e}")


class PersonDetector:
    def __init__(self, model_size='n'):
        """
        Inicializa el detector de personas con todos los modelos YOLOv8
        """
        self.model = None
        self.model_type = None  # 'yolov8', 'yolov8-seg', 'opencv', 'dummy'
        self.model_size = model_size
        self.device = 'cpu'

        if TORCH_AVAILABLE:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.setup_model()

    def setup_model(self):
        """Configura el modelo con múltiples métodos de fallback"""
        print("🔄 Configurando detector de personas...")

        # Método 1: Intentar YOLOv8 con ultralytics
        if ULTRALYTICS_AVAILABLE:
            if self.try_yolov8():
                return

        # Método 2: Intentar YOLO con OpenCV (DNN)
        if self.try_opencv_dnn():
            return

        # Método 3: Usar detector dummy para testing
        self.setup_dummy_detector()

    def try_yolov8(self):
        """Intenta cargar YOLOv8 (normal o segmentación)"""
        try:
            # Determinar si es modelo de segmentación
            is_seg_model = '-seg' in self.model_size
            base_model_size = self.model_size.replace('-seg', '')

            print(f"🔄 Intentando cargar YOLOv8-{self.model_size.upper()}...")

            # Buscar modelo en diferentes ubicaciones
            model_filename = f'yolov8{self.model_size}.pt'
            possible_paths = [
                model_filename,
                f'models/{model_filename}',
                f'../models/{model_filename}'
            ]

            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            # Si no existe, descargarlo
            if model_path is None:
                print("📥 Descargando modelo YOLO...")
                model_path = self.download_yolo_model()

            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)

                if is_seg_model:
                    self.model_type = 'yolov8-seg'
                    print(f"✅ YOLOv8-{base_model_size.upper()}-SEG (Segmentación) cargado exitosamente")
                else:
                    self.model_type = 'yolov8'
                    print(f"✅ YOLOv8-{base_model_size.upper()} cargado exitosamente")

                return True
            else:
                print("❌ No se pudo encontrar o descargar el modelo YOLO")
                return False

        except Exception as e:
            print(f"❌ Error cargando YOLOv8: {e}")
            return False

    def try_opencv_dnn(self):
        """Intenta cargar YOLO con OpenCV DNN"""
        try:
            print("🔄 Intentando cargar YOLO con OpenCV DNN...")

            # Descargar archivos YOLO para OpenCV
            config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
            weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"

            config_path = "models/yolov3-tiny.cfg"
            weights_path = "models/yolov3-tiny.weights"

            # Descargar si no existen
            if not os.path.exists(config_path):
                urllib.request.urlretrieve(config_url, config_path)
            if not os.path.exists(weights_path):
                urllib.request.urlretrieve(weights_url, weights_path)

            if os.path.exists(config_path) and os.path.exists(weights_path):
                self.model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
                self.model_type = 'opencv_dnn'

                # Configurar backend de OpenCV
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

                print("✅ YOLO con OpenCV DNN cargado exitosamente")
                return True
            else:
                print("❌ No se pudieron descargar los archivos YOLO para OpenCV")
                return False

        except Exception as e:
            print(f"❌ Error cargando YOLO con OpenCV: {e}")
            return False

    def setup_dummy_detector(self):
        """Configura un detector dummy para testing"""
        print("⚠️ Usando detector dummy para testing")
        self.model_type = 'dummy'
        print("✅ Detector dummy configurado - mostrará datos de ejemplo")

    def download_yolo_model(self):
        """Descarga el modelo YOLO manualmente"""
        try:
            models_dir = "models"
            os.makedirs(models_dir, exist_ok=True)

            model_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8{self.model_size}.pt"
            model_path = os.path.join(models_dir, f'yolov8{self.model_size}.pt')

            print(f"📥 Descargando {model_url}...")
            urllib.request.urlretrieve(model_url, model_path)

            if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
                print(f"✅ Modelo descargado: {model_path}")
                return model_path
            else:
                print("❌ El archivo descargado es demasiado pequeño")
                return None

        except Exception as e:
            print(f"❌ Error descargando modelo: {e}")
            return None

    def process_video(self, video_path, confidence_thresh=0.5, roi_enabled=False, roi_coords=None, frame_skip=3):
        """
        Procesa un video para detectar personas
        """
        print(f"🎬 Procesando video: {video_path}")

        # Validaciones iniciales
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"El archivo {video_path} no existe")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("❌ No se pudo abrir el video")

        # Obtener propiedades del video
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📊 Video: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")
        if roi_enabled and roi_coords:
            print(f"🎯 ROI activo: {roi_coords}")

        # Variables para el análisis
        frame_count = 0
        processed_frames = 0
        timeline_data = []

        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Procesar cada N frames
                if frame_count % frame_skip == 0:
                    # Aplicar ROI si está habilitado
                    processing_frame = frame.copy()
                    if roi_enabled and roi_coords:
                        x1, y1, x2, y2 = roi_coords
                        # Asegurar que las coordenadas estén dentro de los límites
                        x1 = max(0, min(x1, width))
                        y1 = max(0, min(y1, height))
                        x2 = max(0, min(x2, width))
                        y2 = max(0, min(y2, height))

                        if x2 > x1 and y2 > y1:
                            processing_frame = processing_frame[y1:y2, x1:x2]

                    # Detectar personas según el método disponible
                    if self.model_type == 'yolov8':
                        detections = self.detect_yolov8(processing_frame, confidence_thresh)
                    elif self.model_type == 'yolov8-seg':
                        detections = self.detect_yolov8_seg(processing_frame, confidence_thresh)
                    elif self.model_type == 'opencv_dnn':
                        detections = self.detect_opencv_dnn(processing_frame, confidence_thresh)
                    else:  # dummy
                        detections = self.detect_dummy(processing_frame, frame_count)

                    # Guardar información del frame
                    frame_info = {
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps,
                        'timestamp_formatted': self.format_timestamp(frame_count / fps),
                        'people_count': len(detections),
                        'detections': detections,
                        'confidence_avg': np.mean([d['confidence'] for d in detections]) if detections else 0,
                        'roi_applied': roi_enabled
                    }

                    timeline_data.append(frame_info)
                    processed_frames += 1

                frame_count += 1

                # Mostrar progreso
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    current_count = len(detections) if 'detections' in locals() else 0
                    print(f"📈 Progreso: {progress:.1f}% - Personas: {current_count}")

        except Exception as e:
            print(f"❌ Error durante el procesamiento: {e}")
        finally:
            cap.release()

        processing_time = time.time() - start_time

        # Generar resultados
        return {
            'timeline': timeline_data,
            'summary_stats': self.generate_summary_stats(timeline_data),
            'video_info': {
                'fps': fps,
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'resolution': f"{width}x{height}",
                'duration': total_frames / fps if fps > 0 else 0
            },
            'processing_info': {
                'processing_time': processing_time,
                'model_type': self.model_type,
                'model_size': self.model_size,
                'confidence_threshold': confidence_thresh,
                'frame_skip': frame_skip,
                'roi_enabled': roi_enabled,
                'roi_coords': roi_coords if roi_enabled else None,
                'device': self.device
            }
        }

    def detect_yolov8(self, frame, confidence_thresh):
        """Detección con YOLOv8 normal"""
        try:
            results = self.model(frame, conf=confidence_thresh, verbose=False)
            detections = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls) == 0 and box.conf >= confidence_thresh:  # Persona
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf)

                            detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': confidence,
                                'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                                'area': float((x2 - x1) * (y2 - y1)),
                                'type': 'detection'
                            })

            return detections

        except Exception as e:
            print(f"❌ Error en detección YOLOv8: {e}")
            return []

    def detect_yolov8_seg(self, frame, confidence_thresh):
        """Detección con YOLOv8 Segmentación"""
        try:
            results = self.model(frame, conf=confidence_thresh, verbose=False)
            detections = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        if int(box.cls) == 0 and box.conf >= confidence_thresh:  # Persona
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf)

                            # Obtener máscara de segmentación si está disponible
                            mask = None
                            if hasattr(result, 'masks') and result.masks is not None:
                                masks = result.masks
                                if i < len(masks.data):
                                    mask = masks.data[i].cpu().numpy()

                            detection = {
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': confidence,
                                'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                                'area': float((x2 - x1) * (y2 - y1)),
                                'type': 'segmentation'
                            }

                            if mask is not None:
                                detection['mask'] = mask

                            detections.append(detection)

            return detections

        except Exception as e:
            print(f"❌ Error en detección YOLOv8-Seg: {e}")
            return []

    def detect_opencv_dnn(self, frame, confidence_thresh):
        """Detección con OpenCV DNN"""
        try:
            # Configuración para YOLO
            ln = self.model.getLayerNames()
            ln = [ln[i - 1] for i in self.model.getUnconnectedOutLayers()]

            # Preprocesamiento
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            self.model.setInput(blob)
            outputs = self.model.forward(ln)

            detections = []
            h, w = frame.shape[:2]

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Filtrar personas (clase 0) y confianza
                    if class_id == 0 and confidence > confidence_thresh:
                        box = detection[0:4] * np.array([w, h, w, h])
                        (centerX, centerY, width, height) = box.astype("int")

                        x1 = int(centerX - (width / 2))
                        y1 = int(centerY - (height / 2))
                        x2 = int(centerX + (width / 2))
                        y2 = int(centerY + (height / 2))

                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'center': [float(centerX), float(centerY)],
                            'area': float(width * height),
                            'type': 'detection'
                        })

            return detections

        except Exception as e:
            print(f"❌ Error en detección OpenCV DNN: {e}")
            return []

    def detect_dummy(self, frame, frame_count):
        """Detección dummy para testing"""
        # Simular detecciones para testing
        import random
        detections = []

        # Generar 0-3 detecciones aleatorias
        num_detections = random.randint(0, 3)

        for i in range(num_detections):
            h, w = frame.shape[:2]
            size = random.randint(50, 150)
            x1 = random.randint(0, w - size)
            y1 = random.randint(0, h - size)
            x2 = x1 + size
            y2 = y1 + size

            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': random.uniform(0.5, 0.9),
                'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                'area': float(size * size),
                'type': 'dummy'
            })

        return detections

    def format_timestamp(self, seconds):
        """Formatea segundos a HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def generate_summary_stats(self, timeline_data):
        """Genera estadísticas del análisis"""
        if not timeline_data:
            return {}

        people_counts = [frame['people_count'] for frame in timeline_data]
        confidences = [frame['confidence_avg'] for frame in timeline_data if frame['confidence_avg'] > 0]

        return {
            'total_people_detected': sum(people_counts),
            'max_people': max(people_counts) if people_counts else 0,
            'min_people': min(people_counts) if people_counts else 0,
            'avg_people': np.mean(people_counts) if people_counts else 0,
            'std_people': np.std(people_counts) if people_counts else 0,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'total_frames_processed': len(timeline_data)
        }