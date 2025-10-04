<<<<<<< HEAD
import os
import urllib.request
import torch


def download_yolo_models():
    """Descarga manualmente los modelos YOLOv8"""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # URLs de los modelos YOLOv8
    models = {
        'n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
        's': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
        'm': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
        'l': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
        'x': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt'
    }

    for size, url in models.items():
        model_path = os.path.join(models_dir, f'yolov8{size}.pt')
        if not os.path.exists(model_path):
            print(f"📥 Descargando yolov8{size}.pt...")
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"✅ yolov8{size}.pt descargado exitosamente")
            except Exception as e:
                print(f"❌ Error descargando yolov8{size}.pt: {e}")
        else:
            print(f"✅ yolov8{size}.pt ya existe")


if __name__ == "__main__":
=======
import os
import urllib.request
import torch


def download_yolo_models():
    """Descarga manualmente los modelos YOLOv8"""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # URLs de los modelos YOLOv8
    models = {
        'n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
        's': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
        'm': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
        'l': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
        'x': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt'
    }

    for size, url in models.items():
        model_path = os.path.join(models_dir, f'yolov8{size}.pt')
        if not os.path.exists(model_path):
            print(f"📥 Descargando yolov8{size}.pt...")
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"✅ yolov8{size}.pt descargado exitosamente")
            except Exception as e:
                print(f"❌ Error descargando yolov8{size}.pt: {e}")
        else:
            print(f"✅ yolov8{size}.pt ya existe")


if __name__ == "__main__":
>>>>>>> de9629d342de615f03e0a042d37f165b817b3a5f
    download_yolo_models()