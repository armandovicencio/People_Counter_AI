<<<<<<< HEAD
import streamlit as st
import cv2
import os
import tempfile
import numpy as np
import pandas as pd
import plotly.express as px

# --- Configuración del Path para Módulos Locales ---
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# --- Importación de Módulos Propios ---
try:
    from tracker import ObjectTracker
    from video_loader import load_video_source
except ImportError as e:
    st.error(f"❌ Error al importar módulos locales: {e}")
    st.info("Asegúrate de que la carpeta 'utils' exista y contenga los archivos 'tracker.py' y 'video_loader.py'.")
    st.stop()

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Plataforma de Analítica Visual con IA",
    page_icon="👁️",
    layout="wide",
)


# --- Clase Principal de la Aplicación ---
class VisionAnalyticsApp:
    def __init__(self):
        # --- Inicialización del Estado de la Sesión ---
        if 'tracker' not in st.session_state:
            model_path = 'yolov8n.pt'
            with st.spinner(f"Cargando modelo '{model_path}'..."):
                try:
                    st.session_state.tracker = ObjectTracker(model_path)
                except Exception as e:
                    st.error(f"No se pudo cargar el modelo. Asegúrate de que '{model_path}' existe.")
                    st.stop()

        if 'analysis_type' not in st.session_state:
            st.session_state.analysis_type = "Conteo en Región"
        if 'region_points' not in st.session_state:
            st.session_state.region_points = []
        if 'line_points' not in st.session_state:
            st.session_state.line_points = []
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        # --- CORRECCIÓN: Guardar el tipo de análisis que se ejecutó ---
        if 'last_run_analysis_type' not in st.session_state:
            st.session_state.last_run_analysis_type = None

    def main(self):
        st.title("👁️ Plataforma de Analítica Visual con IA")
        st.markdown("Una plataforma avanzada para conteo de objetos y análisis de video usando YOLOv8.")

        self.setup_sidebar()
        self.setup_main_area()

    def setup_sidebar(self):
        with st.sidebar:
            st.header("⚙️ Configuración")

            source_type = st.selectbox(
                "Selecciona la Fuente del Video",
                ["Subir Archivo", "URL de YouTube", "Stream RTSP"]
            )

            source_path = None
            if source_type == "Subir Archivo":
                source_path = st.file_uploader("📹 Sube un Video", type=['mp4', 'avi', 'mov', 'mkv'])
            elif source_type == "URL de YouTube":
                source_path = st.text_input("🔗 Pega la URL de YouTube")
            elif source_type == "Stream RTSP":
                source_path = st.text_input("📡 Pega la URL del Stream RTSP", "rtsp://...")

            st.markdown("---")

            st.session_state.analysis_type = st.radio(
                "Selecciona el Tipo de Análisis",
                ["Conteo en Región", "Cruce de Línea"],
                key="analysis_type_radio",
                help="Elige qué tipo de análisis realizar."
            )

            st.header("🎯 Definir Zonas")
            st.info("Esta sección se volverá interactiva en una futura actualización.")

            if st.session_state.analysis_type == "Conteo en Región":
                st.write("Define un polígono para el conteo.")
                if not st.session_state.region_points:
                    st.session_state.region_points = [[100, 100], [500, 100], [500, 500], [100, 500]]
                st.write(f"Región Actual (píxeles): {st.session_state.region_points}")
            elif st.session_state.analysis_type == "Cruce de Línea":
                st.write("Define una línea para contar cruces.")
                if not st.session_state.line_points:
                    st.session_state.line_points = [[200, 100], [200, 500]]
                st.write(f"Línea Actual (píxeles): {st.session_state.line_points}")

            st.markdown("---")

            if source_path:
                if st.button("🚀 Iniciar Análisis", use_container_width=True, type="primary"):
                    st.session_state.analysis_results = None
                    # --- CORRECCIÓN: Pasar el tipo de análisis a la función ---
                    self.run_analysis(source_type, source_path, st.session_state.analysis_type)

    def setup_main_area(self):
        st.header("📊 Dashboard de Resultados")
        if st.session_state.analysis_results is not None:
            self.display_results(st.session_state.analysis_results)
        else:
            st.info("Sube un video desde la barra lateral y comienza el análisis para ver los resultados aquí.")

    def run_analysis(self, source_type, source_path, analysis_type):
        # --- CORRECCIÓN: Guardar el tipo de análisis que estamos ejecutando ---
        st.session_state.last_run_analysis_type = analysis_type

        cap, temp_file_path = load_video_source(source_type, source_path)
        if cap is None:
            return

        results_timeline = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        pbar = st.progress(0, text="Inicializando análisis...")
        st_frame_display = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = st.session_state.tracker.process_frame(frame)
            annotated_frame = results.plot()

            frame_data = {'frame': frame_count, 'total_tracks': len(results.boxes)}

            # --- La lógica ahora usa la variable local 'analysis_type' ---
            if analysis_type == "Conteo en Región":
                region = np.array(st.session_state.region_points, dtype=np.int32)
                cv2.polylines(annotated_frame, [region], isClosed=True, color=(0, 255, 0), thickness=3)

                people_in_region = st.session_state.tracker.count_objects_in_region(results, region)
                count = len(people_in_region)
                frame_data['count'] = count
                cv2.putText(annotated_frame, f'En Region: {count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (0, 255, 0), 3)

            elif analysis_type == "Cruce de Línea":
                line = st.session_state.line_points
                p1, p2 = tuple(line[0]), tuple(line[1])
                cv2.line(annotated_frame, p1, p2, (255, 0, 0), 3)

                count_in, count_out = st.session_state.tracker.count_line_crossings(results, (p1, p2))
                frame_data['in'] = count_in
                frame_data['out'] = count_out
                cv2.putText(annotated_frame, f'Entradas: {count_in} | Salidas: {count_out}', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

            results_timeline.append(frame_data)
            frame_count += 1

            progress_percent = (frame_count / total_frames) if total_frames > 0 else 0
            pbar.progress(progress_percent, text=f"Procesando Frame {frame_count} / {total_frames or 'Stream'}")
            st_frame_display.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

        pbar.success("¡Análisis completado!")
        st.session_state.analysis_results = pd.DataFrame(results_timeline).fillna(0)
        st.rerun()

    def display_results(self, df):
        st.subheader("Resultados del Análisis")

        # --- CORRECCIÓN: Usar el tipo de análisis que se ejecutó para la visualización ---
        analysis_type_to_display = st.session_state.last_run_analysis_type

        if analysis_type_to_display == "Conteo en Región":
            # Comprobación de seguridad
            if 'count' in df.columns:
                st.metric("Máximo de Personas en Región", int(df['count'].max()))
                st.metric("Promedio de Personas en Región", round(df['count'].mean(), 2))

                fig = px.line(df, x='frame', y='count', title="Conteo de Personas en la Región a lo Largo del Tiempo")
                fig.update_layout(xaxis_title="Número de Frame", yaxis_title="Cantidad de Personas")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "Los datos de resultados no corresponden a un análisis de 'Conteo en Región'. Por favor, ejecuta el análisis de nuevo.")

        elif analysis_type_to_display == "Cruce de Línea":
            # Comprobación de seguridad
            if 'in' in df.columns and 'out' in df.columns:
                df['total_in'] = df['in'].cumsum()
                df['total_out'] = df['out'].cumsum()

                col1, col2 = st.columns(2)
                col1.metric("Total de Entradas", int(df['total_in'].iloc[-1]))
                col2.metric("Total de Salidas", int(df['total_out'].iloc[-1]))

                fig = px.line(df, x='frame', y=['total_in', 'total_out'],
                              title="Cruces de Línea Acumulados a lo Largo del Tiempo")
                fig.update_layout(xaxis_title="Número de Frame", yaxis_title="Cantidad Acumulada")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "Los datos de resultados no corresponden a un análisis de 'Cruce de Línea'. Por favor, ejecuta el análisis de nuevo.")

        with st.expander("Ver datos crudos"):
            st.dataframe(df)


if __name__ == "__main__":
    app = VisionAnalyticsApp()
    app.main()

=======
import streamlit as st
import cv2
import os
import tempfile
import numpy as np
import pandas as pd
import plotly.express as px

# --- Configuración del Path para Módulos Locales ---
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# --- Importación de Módulos Propios ---
try:
    from tracker import ObjectTracker
    from video_loader import load_video_source
except ImportError as e:
    st.error(f"❌ Error al importar módulos locales: {e}")
    st.info("Asegúrate de que la carpeta 'utils' exista y contenga los archivos 'tracker.py' y 'video_loader.py'.")
    st.stop()

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Plataforma de Analítica Visual con IA",
    page_icon="👁️",
    layout="wide",
)


# --- Clase Principal de la Aplicación ---
class VisionAnalyticsApp:
    def __init__(self):
        # --- Inicialización del Estado de la Sesión ---
        if 'tracker' not in st.session_state:
            model_path = 'yolov8n.pt'
            with st.spinner(f"Cargando modelo '{model_path}'..."):
                try:
                    st.session_state.tracker = ObjectTracker(model_path)
                except Exception as e:
                    st.error(f"No se pudo cargar el modelo. Asegúrate de que '{model_path}' existe.")
                    st.stop()

        if 'analysis_type' not in st.session_state:
            st.session_state.analysis_type = "Conteo en Región"
        if 'region_points' not in st.session_state:
            st.session_state.region_points = []
        if 'line_points' not in st.session_state:
            st.session_state.line_points = []
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        # --- CORRECCIÓN: Guardar el tipo de análisis que se ejecutó ---
        if 'last_run_analysis_type' not in st.session_state:
            st.session_state.last_run_analysis_type = None

    def main(self):
        st.title("👁️ Plataforma de Analítica Visual con IA")
        st.markdown("Una plataforma avanzada para conteo de objetos y análisis de video usando YOLOv8.")

        self.setup_sidebar()
        self.setup_main_area()

    def setup_sidebar(self):
        with st.sidebar:
            st.header("⚙️ Configuración")

            source_type = st.selectbox(
                "Selecciona la Fuente del Video",
                ["Subir Archivo", "URL de YouTube", "Stream RTSP"]
            )

            source_path = None
            if source_type == "Subir Archivo":
                source_path = st.file_uploader("📹 Sube un Video", type=['mp4', 'avi', 'mov', 'mkv'])
            elif source_type == "URL de YouTube":
                source_path = st.text_input("🔗 Pega la URL de YouTube")
            elif source_type == "Stream RTSP":
                source_path = st.text_input("📡 Pega la URL del Stream RTSP", "rtsp://...")

            st.markdown("---")

            st.session_state.analysis_type = st.radio(
                "Selecciona el Tipo de Análisis",
                ["Conteo en Región", "Cruce de Línea"],
                key="analysis_type_radio",
                help="Elige qué tipo de análisis realizar."
            )

            st.header("🎯 Definir Zonas")
            st.info("Esta sección se volverá interactiva en una futura actualización.")

            if st.session_state.analysis_type == "Conteo en Región":
                st.write("Define un polígono para el conteo.")
                if not st.session_state.region_points:
                    st.session_state.region_points = [[100, 100], [500, 100], [500, 500], [100, 500]]
                st.write(f"Región Actual (píxeles): {st.session_state.region_points}")
            elif st.session_state.analysis_type == "Cruce de Línea":
                st.write("Define una línea para contar cruces.")
                if not st.session_state.line_points:
                    st.session_state.line_points = [[200, 100], [200, 500]]
                st.write(f"Línea Actual (píxeles): {st.session_state.line_points}")

            st.markdown("---")

            if source_path:
                if st.button("🚀 Iniciar Análisis", use_container_width=True, type="primary"):
                    st.session_state.analysis_results = None
                    # --- CORRECCIÓN: Pasar el tipo de análisis a la función ---
                    self.run_analysis(source_type, source_path, st.session_state.analysis_type)

    def setup_main_area(self):
        st.header("📊 Dashboard de Resultados")
        if st.session_state.analysis_results is not None:
            self.display_results(st.session_state.analysis_results)
        else:
            st.info("Sube un video desde la barra lateral y comienza el análisis para ver los resultados aquí.")

    def run_analysis(self, source_type, source_path, analysis_type):
        # --- CORRECCIÓN: Guardar el tipo de análisis que estamos ejecutando ---
        st.session_state.last_run_analysis_type = analysis_type

        cap, temp_file_path = load_video_source(source_type, source_path)
        if cap is None:
            return

        results_timeline = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        pbar = st.progress(0, text="Inicializando análisis...")
        st_frame_display = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = st.session_state.tracker.process_frame(frame)
            annotated_frame = results.plot()

            frame_data = {'frame': frame_count, 'total_tracks': len(results.boxes)}

            # --- La lógica ahora usa la variable local 'analysis_type' ---
            if analysis_type == "Conteo en Región":
                region = np.array(st.session_state.region_points, dtype=np.int32)
                cv2.polylines(annotated_frame, [region], isClosed=True, color=(0, 255, 0), thickness=3)

                people_in_region = st.session_state.tracker.count_objects_in_region(results, region)
                count = len(people_in_region)
                frame_data['count'] = count
                cv2.putText(annotated_frame, f'En Region: {count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (0, 255, 0), 3)

            elif analysis_type == "Cruce de Línea":
                line = st.session_state.line_points
                p1, p2 = tuple(line[0]), tuple(line[1])
                cv2.line(annotated_frame, p1, p2, (255, 0, 0), 3)

                count_in, count_out = st.session_state.tracker.count_line_crossings(results, (p1, p2))
                frame_data['in'] = count_in
                frame_data['out'] = count_out
                cv2.putText(annotated_frame, f'Entradas: {count_in} | Salidas: {count_out}', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

            results_timeline.append(frame_data)
            frame_count += 1

            progress_percent = (frame_count / total_frames) if total_frames > 0 else 0
            pbar.progress(progress_percent, text=f"Procesando Frame {frame_count} / {total_frames or 'Stream'}")
            st_frame_display.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

        pbar.success("¡Análisis completado!")
        st.session_state.analysis_results = pd.DataFrame(results_timeline).fillna(0)
        st.rerun()

    def display_results(self, df):
        st.subheader("Resultados del Análisis")

        # --- CORRECCIÓN: Usar el tipo de análisis que se ejecutó para la visualización ---
        analysis_type_to_display = st.session_state.last_run_analysis_type

        if analysis_type_to_display == "Conteo en Región":
            # Comprobación de seguridad
            if 'count' in df.columns:
                st.metric("Máximo de Personas en Región", int(df['count'].max()))
                st.metric("Promedio de Personas en Región", round(df['count'].mean(), 2))

                fig = px.line(df, x='frame', y='count', title="Conteo de Personas en la Región a lo Largo del Tiempo")
                fig.update_layout(xaxis_title="Número de Frame", yaxis_title="Cantidad de Personas")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "Los datos de resultados no corresponden a un análisis de 'Conteo en Región'. Por favor, ejecuta el análisis de nuevo.")

        elif analysis_type_to_display == "Cruce de Línea":
            # Comprobación de seguridad
            if 'in' in df.columns and 'out' in df.columns:
                df['total_in'] = df['in'].cumsum()
                df['total_out'] = df['out'].cumsum()

                col1, col2 = st.columns(2)
                col1.metric("Total de Entradas", int(df['total_in'].iloc[-1]))
                col2.metric("Total de Salidas", int(df['total_out'].iloc[-1]))

                fig = px.line(df, x='frame', y=['total_in', 'total_out'],
                              title="Cruces de Línea Acumulados a lo Largo del Tiempo")
                fig.update_layout(xaxis_title="Número de Frame", yaxis_title="Cantidad Acumulada")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "Los datos de resultados no corresponden a un análisis de 'Cruce de Línea'. Por favor, ejecuta el análisis de nuevo.")

        with st.expander("Ver datos crudos"):
            st.dataframe(df)


if __name__ == "__main__":
    app = VisionAnalyticsApp()
    app.main()

>>>>>>> de9629d342de615f03e0a042d37f165b817b3a5f
