import cv2
import pytesseract
from PIL import Image
import easyocr
import re
from datetime import datetime
import exifread
import os


class MetadataExtractor:
    def __init__(self):
        """Inicializa el extractor de metadatos"""
        self.reader = easyocr.Reader(['en'])  # Para OCR en inglés
        self.date_patterns = [
            r'\d{4}[-/]\d{2}[-/]\d{2}',  # YYYY-MM-DD
            r'\d{2}[-/]\d{2}[-/]\d{4}',  # DD-MM-YYYY
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'  # Variaciones
        ]
        self.time_patterns = [
            r'\d{1,2}:\d{2}:\d{2}',  # HH:MM:SS
            r'\d{1,2}:\d{2}',  # HH:MM
            r'\d{1,2}:\d{2}[ap]m'  # HH:MMam/pm
        ]
        self.gps_patterns = [
            r'[-+]?\d{1,3}\.\d+\s*[,;]\s*[-+]?\d{1,3}\.\d+',  # Lat, Lon
            r'Lat:\s*[-+]?\d{1,3}\.\d+\s*Lon:\s*[-+]?\d{1,3}\.\d+',  # Lat: X Lon: Y
            r'N\s*\d{1,3}\.\d+\s*W\s*\d{1,3}\.\d+'  # N X W Y
        ]

    def extract_basic_metadata(self, video_path):
        """
        Extrae metadatos básicos del archivo de video

        Args:
            video_path: Ruta al archivo de video

        Returns:
            Dict con metadatos básicos
        """
        try:
            cap = cv2.VideoCapture(video_path)

            metadata = {
                'filename': os.path.basename(video_path),
                'file_size': f"{os.path.getsize(video_path) / (1024 * 1024):.2f} MB",
                'duration': None,
                'fps': None,
                'resolution': None,
                'frame_count': None,
                'creation_time': None,
                'gps_data': None,
                'ocr_data': None
            }

            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                metadata.update({
                    'fps': fps,
                    'frame_count': frame_count,
                    'resolution': f"{width}x{height}",
                    'duration': f"{frame_count / fps:.2f} segundos" if fps > 0 else "Desconocido"
                })

            cap.release()

            # Intentar extraer más metadatos del archivo
            metadata.update(self.extract_file_metadata(video_path))

            # Procesar algunos frames para OCR
            metadata['ocr_data'] = self.extract_ocr_from_video(video_path)

            return metadata

        except Exception as e:
            print(f"Error extrayendo metadatos básicos: {e}")
            return {'error': str(e)}

    def extract_file_metadata(self, file_path):
        """
        Extrae metadatos del archivo usando diferentes métodos
        """
        metadata = {}

        try:
            # Información del archivo
            stat = os.stat(file_path)
            metadata['file_created'] = datetime.fromtimestamp(stat.st_ctime).isoformat()
            metadata['file_modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()

        except Exception as e:
            print(f"Error extrayendo metadatos del archivo: {e}")

        return metadata

    def extract_ocr_from_video(self, video_path, sample_frames=10):
        """
        Extrae texto de frames del video usando OCR

        Args:
            video_path: Ruta al video
            sample_frames: Número de frames a muestrear

        Returns:
            Dict con texto extraído y metadatos encontrados
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                return {'error': 'Video vacío o corrupto'}

            # Muestrear frames distribuidos
            frame_indices = [int(i * total_frames / sample_frames) for i in range(sample_frames)]

            all_text = []
            found_metadata = {
                'dates': set(),
                'times': set(),
                'coordinates': set()
            }

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    # Procesar frame con OCR
                    frame_text = self.process_frame_ocr(frame)
                    all_text.extend(frame_text)

                    # Buscar patrones en el texto
                    for text in frame_text:
                        self.extract_patterns_from_text(text, found_metadata)

            cap.release()

            return {
                'sample_text': list(set(all_text))[:20],  # Limitar cantidad
                'extracted_metadata': {
                    'dates': list(found_metadata['dates']),
                    'times': list(found_metadata['times']),
                    'coordinates': list(found_metadata['coordinates'])
                }
            }

        except Exception as e:
            return {'error': f"Error en OCR: {str(e)}"}

    def process_frame_ocr(self, frame):
        """
        Procesa un frame con OCR

        Args:
            frame: Frame de video

        Returns:
            Lista de textos detectados
        """
        try:
            # Preprocesar imagen para mejor OCR
            processed_frame = self.preprocess_frame_for_ocr(frame)

            # Usar EasyOCR
            results = self.reader.readtext(processed_frame, detail=0)

            # Filtrar textos válidos
            valid_texts = []
            for text in results:
                if len(text.strip()) > 3:  # Ignorar textos muy cortos
                    valid_texts.append(text.strip())

            return valid_texts

        except Exception as e:
            print(f"Error en OCR de frame: {e}")
            return []

    def preprocess_frame_for_ocr(self, frame):
        """
        Preprocesa el frame para mejorar resultados de OCR

        Args:
            frame: Frame original

        Returns:
            Frame procesado
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar filtros para mejorar contraste
        # ...

        return gray

    def extract_patterns_from_text(self, text, found_metadata):
        """
        Extrae patrones de fecha, hora y coordenadas del texto

        Args:
            text: Texto a analizar
            found_metadata: Dict para almacenar metadatos encontrados
        """
        # Buscar fechas
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                found_metadata['dates'].add(match)

        # Buscar horas
        for pattern in self.time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                found_metadata['times'].add(match)

        # Buscar coordenadas GPS
        for pattern in self.gps_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                found_metadata['coordinates'].add(match)