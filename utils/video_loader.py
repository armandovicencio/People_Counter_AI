import cv2
import yt_dlp
import streamlit as st
import tempfile
import os

@st.cache_data
def get_youtube_stream_url(youtube_url):
    """
    Obtiene la URL del stream de video directo de una URL de YouTube.
    Usa el caché de Streamlit para no descargar la información repetidamente.
    """
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info['url']
    except Exception as e:
        st.error(f"❌ Error al obtener el video de YouTube: {e}")
        return None

def load_video_source(source_type, source_path):
    """
    Crea un objeto VideoCapture de OpenCV a partir de diferentes fuentes.

    Args:
        source_type (str): El tipo de fuente ('Subir Archivo', 'Stream RTSP', 'URL de YouTube').
        source_path: La ruta del archivo, la URL RTSP o la URL de YouTube.

    Returns:
        Un objeto cv2.VideoCapture y la ruta del archivo temporal (si aplica).
    """
    temp_file_path = None # Para manejar archivos subidos

    # --- CORRECCIÓN APLICADA AQUÍ ---
    # Ahora las condiciones coinciden con las opciones de la interfaz en app.py
    if source_type == 'Subir Archivo':
        # Crea un archivo temporal para que OpenCV pueda leerlo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(source_path.getvalue())
            temp_file_path = tmp_file.name
        cap = cv2.VideoCapture(temp_file_path)

    elif source_type == 'Stream RTSP':
        # Conecta directamente a la URL de RTSP
        cap = cv2.VideoCapture(source_path)

    elif source_type == 'URL de YouTube':
        # Obtiene la URL del stream y la pasa a OpenCV
        stream_url = get_youtube_stream_url(source_path)
        if stream_url:
            cap = cv2.VideoCapture(stream_url)
        else:
            return None, None

    else:
        # Este error ya no debería aparecer.
        st.error(f"Tipo de fuente no válido: '{source_type}'")
        return None, None

    if not cap.isOpened():
        st.error(f"❌ No se pudo abrir la fuente de video: {source_path}")
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path) # Limpia el archivo temporal si falla
        return None, None

    return cap, temp_file_path
