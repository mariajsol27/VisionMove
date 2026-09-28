# ============================================================
# VISIONMOVE MOBILE - OPTIMIZADO PARA STREAMLIT CLOUD
# Cámara en vivo + YOLO + distancia + voz + accesibilidad
# Proyecto académico - Ingeniería Biomédica
# ============================================================

import time
import threading
import hashlib

import av
import cv2
import streamlit as st
import streamlit.components.v1 as components

from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
)


# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================

st.set_page_config(
    page_title="VisionMove",
    page_icon="👁️",
    layout="centered",
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

MODEL_NAME = "yolov8n.pt"

CONFIDENCE = 0.40

# Cada cuántos segundos actualizar el mensaje de detección
ANNOUNCE_EVERY = 3.0

# IMPORTANTE:
# Cambia este valor por el resultado de tu calibración móvil.
FOCAL_LENGTH_PX = 700.0

# Procesar solo 1 de cada N frames.
PROCESS_EVERY_N_FRAMES = 3

# Resolución usada internamente por YOLO.
YOLO_IMAGE_SIZE = 320


# ============================================================
# ANCHOS REALES APROXIMADOS EN METROS
# ============================================================

ANCHOS_REALES = {
    "person": 0.45,
    "chair": 0.45,
    "couch": 1.80,
    "dining table": 1.20,
    "bench": 1.20,
    "backpack": 0.30,
    "suitcase": 0.45,
    "bottle": 0.07,
    "cell phone": 0.075,
    "laptop": 0.35,
    "book": 0.20,
    "dog": 0.35,
    "cat": 0.25,
    "bicycle": 0.60,
    "motorcycle": 0.80,
    "car": 1.80,
    "bus": 2.50,
    "truck": 2.50,
    "tv": 0.90,
}


# ============================================================
# TRADUCCIÓN
# ============================================================

TRADUCCION = {
    "person": "persona",
    "bicycle": "bicicleta",
    "car": "automóvil",
    "motorcycle": "motocicleta",
    "bus": "bus",
    "truck": "camión",
    "chair": "silla",
    "couch": "sofá",
    "bed": "cama",
    "dining table": "mesa",
    "bench": "banca",
    "backpack": "mochila",
    "handbag": "bolso",
    "suitcase": "maleta",
    "umbrella": "paraguas",
    "bottle": "botella",
    "cup": "taza",
    "cell phone": "celular",
    "laptop": "computador",
    "keyboard": "teclado",
    "mouse": "mouse",
    "book": "libro",
    "dog": "perro",
    "cat": "gato",
    "potted plant": "planta",
    "tv": "televisor",
}


# ============================================================
# RIESGO BASE
# ============================================================

RIESGOS = {
    "person": "Medio",
    "chair": "Alto",
    "couch": "Alto",
    "dining table": "Alto",
    "backpack": "Alto",
    "suitcase": "Alto",
    "bicycle": "Alto",
    "motorcycle": "Alto",
    "car": "Alto",
    "bus": "Alto",
    "truck": "Alto",
    "dog": "Medio",
    "cat": "Medio",
    "bed": "Medio",
    "bottle": "Bajo",
    "cell phone": "Bajo",
    "book": "Bajo",
    "laptop": "Bajo",
    "tv": "Bajo",
}


# ============================================================
# CARGAR MODELO
# ============================================================

@st.cache_resource
def cargar_modelo():
    return YOLO(MODEL_NAME)


modelo = cargar_modelo()


# ============================================================
# FUNCIONES DE APOYO
# ============================================================

def determinar_posicion(x_centro, ancho_imagen):
    proporcion = x_centro / ancho_imagen

    if proporcion < 0.33:
        return "a la izquierda"

    if proporcion > 0.66:
        return "a la derecha"

    return "al frente"


def calcular_distancia(nombre, ancho_px):
    if ancho_px <= 0:
        return None

    if nombre not in ANCHOS_REALES:
        return None

    ancho_real = ANCHOS_REALES[nombre]

    distancia = (
        ancho_real *
        FOCAL_LENGTH_PX
    ) / ancho_px

    return max(
        0.20,
        min(distancia, 15.0)
    )


def ajustar_riesgo(riesgo, distancia):
    if distancia is None:
        return riesgo

    if distancia < 1.0:
        if riesgo == "Bajo":
            return "Medio"

        if riesgo == "Medio":
            return "Alto"

    return riesgo


def color_riesgo(riesgo):
    if riesgo == "Alto":
        return (0, 0, 255)

    if riesgo == "Medio":
        return (0, 215, 255)

    return (0, 255, 0)


def ordenar_detecciones(detecciones):
    prioridad = {
        "Alto": 0,
        "Medio": 1,
        "Bajo": 2,
    }

    return sorted(
        detecciones,
        key=lambda objeto: (
            prioridad.get(
                objeto["riesgo"],
                10,
            ),
            (
                objeto["distancia"]
                if objeto["distancia"] is not None
                else 100
            ),
        ),
    )


def objetos_unicos(detecciones):
    resultado = []
    utilizados = set()

    for objeto in ordenar_detecciones(
        detecciones
    ):
        nombre = objeto["nombre_original"]

        if nombre not in utilizados:
            resultado.append(objeto)
            utilizados.add(nombre)

    return resultado


def construir_mensaje(detecciones):
    if not detecciones:
        return None

    objetos = objetos_unicos(
        detecciones
    )[:3]

    frases = []

    for objeto in objetos:
        nombre = objeto["nombre"]
        posicion = objeto["posicion"]
        distancia = objeto["distancia"]
        riesgo = objeto["riesgo"]

        if distancia is not None:
            frase = (
                f"{nombre} {posicion}, "
                f"a aproximadamente "
                f"{distancia:.1f} metros"
            )
        else:
            frase = (
                f"{nombre} {posicion}"
            )

        if riesgo == "Alto":
            frase += ", precaución"

        frases.append(frase)

    return ". ".join(frases) + "."


# ============================================================
# PROCESADOR DE VIDEO
# ============================================================

class VisionMoveProcessor(VideoProcessorBase):

    def __init__(self):
        self.lock = threading.Lock()

        self.latest_message = None
        self.latest_objects = []

        self.last_update = 0.0

        self.frame_counter = 0

        # Guardamos las últimas detecciones para dibujarlas
        # también en frames que no procesamos con YOLO.
        self.last_detections = []


    def recv(self, frame):

        imagen = frame.to_ndarray(
            format="bgr24"
        )

        alto_imagen, ancho_imagen = (
            imagen.shape[:2]
        )

        self.frame_counter += 1

        # ====================================================
        # PROCESAR YOLO SOLO 1 DE CADA 3 FRAMES
        # ====================================================

        if (
            self.frame_counter %
            PROCESS_EVERY_N_FRAMES
            == 0
        ):

            resultados = modelo.predict(
                source=imagen,
                conf=CONFIDENCE,
                imgsz=YOLO_IMAGE_SIZE,
                verbose=False,
            )

            resultado = resultados[0]

            detecciones = []

            for caja in resultado.boxes:

                coordenadas = (
                    caja.xyxy[0]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = coordenadas

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                clase_id = int(
                    caja.cls[0]
                    .cpu()
                    .numpy()
                )

                confianza = float(
                    caja.conf[0]
                    .cpu()
                    .numpy()
                )

                nombre_original = (
                    modelo.names[
                        clase_id
                    ]
                )

                nombre = TRADUCCION.get(
                    nombre_original,
                    nombre_original,
                )

                centro_x = (
                    x1 + x2
                ) / 2

                posicion = determinar_posicion(
                    centro_x,
                    ancho_imagen,
                )

                ancho_objeto_px = (
                    x2 - x1
                )

                distancia = calcular_distancia(
                    nombre_original,
                    ancho_objeto_px,
                )

                riesgo_base = RIESGOS.get(
                    nombre_original,
                    "Bajo",
                )

                riesgo = ajustar_riesgo(
                    riesgo_base,
                    distancia,
                )

                objeto = {
                    "nombre": nombre,
                    "nombre_original": nombre_original,
                    "confianza": confianza,
                    "posicion": posicion,
                    "distancia": distancia,
                    "riesgo": riesgo,
                    "coords": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                }

                detecciones.append(
                    objeto
                )

            self.last_detections = (
                detecciones
            )

            tiempo_actual = time.time()

            if (
                tiempo_actual -
                self.last_update
                >= ANNOUNCE_EVERY
            ):

                mensaje = construir_mensaje(
                    detecciones
                )

                with self.lock:
                    self.latest_message = mensaje
                    self.latest_objects = detecciones

                self.last_update = (
                    tiempo_actual
                )

        # ====================================================
        # DIBUJAR LAS ÚLTIMAS DETECCIONES
        # ====================================================

        for objeto in self.last_detections:

            x1, y1, x2, y2 = (
                objeto["coords"]
            )

            riesgo = objeto["riesgo"]
            distancia = objeto["distancia"]
            nombre = objeto["nombre"]

            color = color_riesgo(
                riesgo
            )

            cv2.rectangle(
                imagen,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            if distancia is not None:
                etiqueta = (
                    f"{nombre} | "
                    f"{distancia:.1f} m"
                )
            else:
                etiqueta = nombre

            cv2.putText(
                imagen,
                etiqueta,
                (
                    x1,
                    max(
                        30,
                        y1 - 10,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        return av.VideoFrame.from_ndarray(
            imagen,
            format="bgr24",
        )


# ============================================================
# ESTILO
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #080808;
        color: white;
    }

    .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 850px;
    }

    h1, h2, h3, p, label {
        color: white !important;
    }

    button {
        min-height: 60px !important;
        font-size: 20px !important;
        font-weight: bold !important;
    }

    div[role="radiogroup"] label {
        background-color: #1a1a1a !important;
        border: 2px solid #555555 !important;
        border-radius: 14px !important;
        padding: 18px 20px !important;
        margin-bottom: 12px !important;
        min-height: 72px !important;
        font-size: 24px !important;
        font-weight: bold !important;
        display: flex !important;
        align-items: center !important;
    }

    div[role="radiogroup"] label:hover {
        border-color: #2196F3 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ENCABEZADO
# ============================================================

st.title(
    "👁️ VisionMove"
)

st.write(
    """
    Asistente experimental para apoyar la identificación
    de objetos y obstáculos mediante cámara y voz.
    """
)


# ============================================================
# MENÚ
# ============================================================

seccion = st.radio(
    "Menú principal",
    [
        "🏠 Inicio",
        "📖 Instrucciones",
        "📷 Reconocimiento",
    ],
)


# ============================================================
# INICIO
# ============================================================

if seccion == "🏠 Inicio":

    st.header(
        "Bienvenido a VisionMove"
    )

    st.write(
        """
        VisionMove utiliza inteligencia artificial
        para identificar objetos presentes en el entorno.
        """
    )

    components.html(
        """
        <button
            onclick="hablarInicio()"
            style="
                width:100%;
                padding:20px;
                font-size:22px;
                background:#1565C0;
                color:white;
                border:none;
                border-radius:14px;
                font-weight:bold;
            "
        >
            🔊 Escuchar instrucciones
        </button>

        <script>
        function hablarInicio() {

            window.speechSynthesis.cancel();

            const mensaje =
                new SpeechSynthesisUtterance(
                    "Bienvenido a VisionMove. "
                    + "Este sistema reconoce objetos del entorno "
                    + "y proporciona información mediante voz. "
                    + "Para comenzar, seleccione Reconocimiento "
                    + "en el menú principal."
                );

            mensaje.lang = "es-CO";
            mensaje.rate = 0.9;
            mensaje.volume = 1.0;

            window.speechSynthesis.speak(
                mensaje
            );
        }
        </script>
        """,
        height=90,
    )

    st.info(
        """
        VisionMove es un prototipo académico.
        No sustituye bastón blanco, perro guía ni técnicas
        profesionales de orientación y movilidad.
        """
    )


# ============================================================
# INSTRUCCIONES
# ============================================================

elif seccion == "📖 Instrucciones":

    st.header(
        "📖 Instrucciones de uso"
    )

    st.markdown(
        """
        **1.** Seleccione **Reconocimiento**.

        **2.** Pulse **START**.

        **3.** Permita el acceso a la cámara.

        **4.** Oriente el teléfono hacia el frente.

        **5.** VisionMove comenzará a reconocer objetos.

        **6.** Escuche la posición y distancia aproximada.
        """
    )

    components.html(
        """
        <button
            onclick="leerInstrucciones()"
            style="
                width:100%;
                padding:20px;
                font-size:22px;
                background:#1565C0;
                color:white;
                border:none;
                border-radius:14px;
                font-weight:bold;
            "
        >
            🔊 Escuchar instrucciones completas
        </button>

        <script>
        function leerInstrucciones() {

            window.speechSynthesis.cancel();

            const mensaje =
                new SpeechSynthesisUtterance(
                    "Instrucciones de VisionMove. "
                    + "Paso uno. Seleccione Reconocimiento. "
                    + "Paso dos. Pulse el botón Start. "
                    + "Paso tres. Permita el acceso a la cámara. "
                    + "Paso cuatro. Oriente la cámara hacia el frente. "
                    + "Paso cinco. VisionMove comenzará a reconocer objetos. "
                    + "El sistema indicará por voz la posición "
                    + "y distancia aproximada."
                );

            mensaje.lang = "es-CO";
            mensaje.rate = 0.9;
            mensaje.volume = 1.0;

            window.speechSynthesis.speak(
                mensaje
            );
        }
        </script>
        """,
        height=90,
    )


# ============================================================
# RECONOCIMIENTO
# ============================================================

elif seccion == "📷 Reconocimiento":

    st.header(
        "📷 Reconocimiento del entorno"
    )

    st.write(
        """
        El botón para comenzar la cámara aparece
        después de la siguiente ayuda.
        """
    )

    components.html(
        """
        <button
            onclick="indicarInicio()"
            style="
                width:100%;
                padding:20px;
                font-size:22px;
                background:#00897B;
                color:white;
                border:none;
                border-radius:14px;
                font-weight:bold;
            "
        >
            🔊 ¿Dónde inicio?
        </button>

        <script>
        function indicarInicio() {

            window.speechSynthesis.cancel();

            const mensaje =
                new SpeechSynthesisUtterance(
                    "El botón para comenzar se encuentra "
                    + "inmediatamente debajo. "
                    + "Deslice hacia abajo y pulse Start. "
                    + "Después permita el acceso a la cámara."
                );

            mensaje.lang = "es-CO";
            mensaje.rate = 0.9;
            mensaje.volume = 1.0;

            window.speechSynthesis.speak(
                mensaje
            );
        }
        </script>
        """,
        height=90,
    )

    st.warning(
        """
        Mantenga el teléfono orientado hacia el frente.
        """
    )


    # ========================================================
    # CÁMARA WEBRTC
    # ========================================================

    ctx = webrtc_streamer(
        key="visionmove-mobile",

        mode=WebRtcMode.SENDRECV,

        video_processor_factory=(
            VisionMoveProcessor
        ),

        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        },

        media_stream_constraints={
            "video": {
                "facingMode": {
                    "ideal": "environment"
                },
                "width": {
                    "ideal": 640
                },
                "height": {
                    "ideal": 480
                },
                "frameRate": {
                    "ideal": 10,
                    "max": 12
                },
            },

            "audio": False,
        },

        async_processing=True,
    )


    # ========================================================
    # SESSION STATE
    # ========================================================

    if (
        "ultimo_mensaje_hablado"
        not in st.session_state
    ):
        st.session_state[
            "ultimo_mensaje_hablado"
        ] = ""


    # ========================================================
    # RESULTADOS
    # ========================================================

    @st.fragment(
        run_every=1.0
    )
    def actualizar_resultado():

        if not ctx.state.playing:

            st.info(
                "Presiona START para iniciar la cámara."
            )

            return


        procesador = (
            ctx.video_processor
        )

        if procesador is None:
            return


        with procesador.lock:

            mensaje = (
                procesador.latest_message
            )

            objetos = list(
                procesador.latest_objects
            )


        if not mensaje:

            st.info(
                "Buscando objetos..."
            )

            return


        # ----------------------------------------------------
        # MENSAJE ACCESIBLE
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div
                role="alert"
                aria-live="assertive"
                aria-atomic="true"
                style="
                    background-color:#162d1d;
                    border:2px solid #25a244;
                    border-radius:12px;
                    padding:18px;
                    margin-top:10px;
                    margin-bottom:10px;
                    color:white;
                    font-size:20px;
                    font-weight:bold;
                "
            >
                {mensaje}
            </div>
            """,
            unsafe_allow_html=True,
        )


        # ----------------------------------------------------
        # VOZ AUTOMÁTICA
        # ----------------------------------------------------

        identificador = hashlib.md5(
            mensaje.encode(
                "utf-8"
            )
        ).hexdigest()


        if (
            identificador
            !=
            st.session_state[
                "ultimo_mensaje_hablado"
            ]
        ):

            mensaje_js = (
                mensaje
                .replace("\\", "\\\\")
                .replace("'", "\\'")
                .replace("\n", " ")
            )

            components.html(
                f"""
                <script>

                window.speechSynthesis.cancel();

                const mensaje =
                    new SpeechSynthesisUtterance(
                        '{mensaje_js}'
                    );

                mensaje.lang = 'es-CO';
                mensaje.rate = 1.0;
                mensaje.volume = 1.0;

                window.speechSynthesis.speak(
                    mensaje
                );

                </script>
                """,
                height=1,
            )

            st.session_state[
                "ultimo_mensaje_hablado"
            ] = identificador


        # ----------------------------------------------------
        # DETALLES
        # ----------------------------------------------------

        if objetos:

            with st.expander(
                "Objetos detectados"
            ):

                for objeto in ordenar_detecciones(
                    objetos
                )[:5]:

                    if (
                        objeto["distancia"]
                        is not None
                    ):

                        st.write(
                            f"**{objeto['nombre'].title()}** — "
                            f"{objeto['posicion']} — "
                            f"{objeto['distancia']:.1f} m — "
                            f"Riesgo {objeto['riesgo']}"
                        )

                    else:

                        st.write(
                            f"**{objeto['nombre'].title()}** — "
                            f"{objeto['posicion']} — "
                            f"Riesgo {objeto['riesgo']}"
                        )


    actualizar_resultado()


# ============================================================
# PIE
# ============================================================

st.divider()

st.caption(
    """
    VisionMove • Prototipo académico de Ingeniería Biomédica
    """
)
