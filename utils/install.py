#!/usr/bin/env python3
"""
Script para descargar el modelo YOLOv8 automáticamente
"""

from ultralytics import YOLO
import os


def download_yolo_model(model_name='yolov8n.pt'):
    """Descarga el modelo YOLOv8 si no existe"""
    print(f"🔍 Verificando modelo {model_name}...")

    if os.path.exists(model_name):
        print(f"✅ Modelo {model_name} ya existe.")
        return True

    print(f"📥 Descargando {model_name}...")
    try:
        model = YOLO(model_name)
        print(f"✅ {model_name} descargado exitosamente!")
        return True
    except Exception as e:
        print(f"❌ Error al descargar {model_name}: {e}")
        return False


if __name__ == "__main__":
    download_yolo_model('yolov8n.pt')