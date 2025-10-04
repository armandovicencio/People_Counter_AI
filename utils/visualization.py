import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class VideoVisualizer:
    def __init__(self):
        """Inicializa el visualizador"""
        self.colors = {
            'person': (0, 255, 0),
            'roi': (255, 0, 0),
            'text': (255, 255, 255)
        }

    def draw_detections(self, frame, detections, show_count=True):
        """
        Dibuja detecciones en un frame

        Args:
            frame: Frame original
            detections: Lista de detecciones
            show_count: Mostrar contador

        Returns:
            Frame con detecciones dibujadas
        """
        frame_with_detections = frame.copy()

        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']

            x1, y1, x2, y2 = map(int, bbox)

            # Dibujar bounding box
            cv2.rectangle(
                frame_with_detections,
                (x1, y1), (x2, y2),
                self.colors['person'],
                2
            )

            # Dibujar etiqueta de confianza
            label = f"Person: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            cv2.rectangle(
                frame_with_detections,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                self.colors['person'],
                -1
            )

            cv2.putText(
                frame_with_detections,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.colors['text'],
                1
            )

        # Mostrar contador total
        if show_count:
            count_text = f"Personas: {len(detections)}"
            cv2.putText(
                frame_with_detections,
                count_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                self.colors['text'],
                2
            )

        return frame_with_detections

    def draw_roi(self, frame, roi_coords):
        """
        Dibuja región de interés en el frame

        Args:
            frame: Frame original
            roi_coords: Coordenadas de la ROI [x1, y1, x2, y2]

        Returns:
            Frame con ROI dibujada
        """
        frame_with_roi = frame.copy()

        x1, y1, x2, y2 = map(int, roi_coords)

        # Dibujar rectángulo de ROI
        cv2.rectangle(
            frame_with_roi,
            (x1, y1), (x2, y2),
            self.colors['roi'],
            2
        )

        # Etiqueta de ROI
        cv2.putText(
            frame_with_roi,
            "ROI",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.colors['roi'],
            2
        )

        return frame_with_roi

    def create_timeline_chart(self, timeline_data, title="Conteo de Personas"):
        """
        Crea gráfico de línea de tiempo

        Args:
            timeline_data: Datos de timeline
            title: Título del gráfico

        Returns:
            Figura de Plotly
        """
        if not timeline_data:
            return None

        # Preparar datos
        timestamps = [item['timestamp'] for item in timeline_data]
        counts = [item['people_count'] for item in timeline_data]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=counts,
            mode='lines+markers',
            name='Personas',
            line=dict(color='green', width=2),
            marker=dict(size=4)
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Tiempo (segundos)',
            yaxis_title='Número de Personas',
            hovermode='x unified',
            showlegend=True
        )

        return fig

    def create_heatmap(self, detections_data, frame_width, frame_height):
        """
        Crea heatmap de densidad de personas

        Args:
            detections_data: Datos de todas las detecciones
            frame_width: Ancho del frame
            frame_height: Alto del frame

        Returns:
            Figura de Plotly con heatmap
        """
        # Extraer todas las posiciones centrales
        all_centers = []

        for frame_data in detections_data:
            for detection in frame_data.get('detections', []):
                center = detection.get('center', [0, 0])
                if center:
                    all_centers.append(center)

        if not all_centers:
            return None

        # Convertir a arrays numpy
        centers_array = np.array(all_centers)

        # Crear heatmap
        fig = px.density_heatmap(
            x=centers_array[:, 0],
            y=centers_array[:, 1],
            title="Mapa de Calor - Densidad de Personas",
            labels={'x': 'Posición X', 'y': 'Posición Y'},
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            xaxis_range=[0, frame_width],
            yaxis_range=[0, frame_height]
        )

        return fig