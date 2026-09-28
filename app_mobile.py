# ============================================================
# VISIONMOVE MOBILE
# Cámara en vivo + YOLO + distancia + voz + menú accesible
# Proyecto académico - Ingeniería Biomédica
# ============================================================

import streamlit as st
import streamlit.components.v1 as components

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode
)

from ultralytics import YOLO

import av
import cv2
import time
import threading
import hashlib


# ============================================================
# CONFIGURACIÓN STREAMLIT
# ============================================================

st.set_page_config(
    page_title="VisionMove",
    page_icon="👁️",
    layout="centered"
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

MODEL_NAME = "yolov8n.pt"

CONFIDENCE = 0.40

# Cada cuántos segundos se genera un nuevo aviso
ANNOUNCE_EVERY = 3.0

# CAMBIA ESTE VALOR POR TU CALIBRACIÓN
FOCAL_LENGTH_PX = 785.0


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

    "tv": 0.90
}


# ============================================================
# TRADUCCIÓN DE OBJETOS
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

    "tv": "televisor"
}


# ============================================================
# RIESGOS BASE
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
    "tv": "Bajo"
}


# ============================================================
# CARGAR YOLO
# ============================================================

@st.cache_resource
def cargar_modelo():

    print("Cargando modelo YOLO...")

    return YOLO(
        MODEL_NAME
    )


modelo = cargar_modelo()


# ============================================================
# POSICIÓN
# ============================================================

def determinar_posicion(
    x_centro,
    ancho_imagen
):

    proporcion = (
        x_centro /
        ancho_imagen
    )

    if proporcion < 0.33:
        return "a la izquierda"

    elif proporcion > 0.66:
        return "a la derecha"

    else:
        return "al frente"


# ============================================================
# DISTANCIA
# ============================================================

def calcular_distancia(
    nombre,
    ancho_px
):

    if ancho_px <= 0:
        return None

    if nombre not in ANCHOS_REALES:
        return None

    ancho_real = (
        ANCHOS_REALES[nombre]
    )

    distancia = (
        ancho_real *
        FOCAL_LENGTH_PX
    ) / ancho_px

    if distancia < 0.20:
        distancia = 0.20

    if distancia > 15:
        distancia = 15

    return distancia


# ============================================================
# AJUSTAR RIESGO
# ============================================================

def ajustar_riesgo(
    riesgo,
    distancia
):

    if distancia is None:
        return riesgo

    if distancia < 1.0:

        if riesgo == "Bajo":
            return "Medio"

        elif riesgo == "Medio":
            return "Alto"

    return riesgo


# ============================================================
# COLOR SEGÚN RIESGO
# ============================================================

def color_riesgo(
    riesgo
):

    if riesgo == "Alto":
        return (0, 0, 255)

    elif riesgo == "Medio":
        return (0, 215, 255)

    return (0, 255, 0)


# ============================================================
# ORDENAR DETECCIONES
# ============================================================

def ordenar_detecciones(
    detecciones
):

    prioridad = {
        "Alto": 0,
        "Medio": 1,
        "Bajo": 2
    }

    return sorted(
        detecciones,

        key=lambda x: (
            prioridad.get(
                x["riesgo"],
                10
            ),

            (
                x["distancia"]
                if x["distancia"] is not None
                else 100
            )
        )
    )


# ============================================================
# ELIMINAR REPETIDOS
# ============================================================

def objetos_unicos(
    detecciones
):

    resultado = []

    utilizados = set()

    for objeto in ordenar_detecciones(
        detecciones
    ):

        nombre = objeto[
            "nombre_original"
        ]

        if nombre not in utilizados:

            resultado.append(
                objeto
            )

            utilizados.add(
                nombre
            )

    return resultado


# ============================================================
# CONSTRUIR MENSAJE
# ============================================================

def construir_mensaje(
    detecciones
):

    if not detecciones:
        return None

    objetos = objetos_unicos(
        detecciones
    )

    # Máximo 3 objetos
    objetos = objetos[:3]

    frases = []

    for objeto in objetos:

        nombre = objeto[
            "nombre"
        ]

        posicion = objeto[
            "posicion"
        ]

        distancia = objeto[
            "distancia"
        ]

        riesgo = objeto[
            "riesgo"
        ]

        if distancia is not None:

            frase = (
                f"{nombre} "
                f"{posicion}, "
                f"a aproximadamente "
                f"{distancia:.1f} metros"
            )

        else:

            frase = (
                f"{nombre} "
                f"{posicion}"
            )

        if riesgo == "Alto":

            frase += (
                ", precaución"
            )

        frases.append(
            frase
        )

    return (
        ". ".join(frases)
        +
        "."
    )


# ============================================================
# PROCESADOR DE VIDEO
# ============================================================

class VisionMoveProcessor(
    VideoProcessorBase
):

    def __init__(self):

        self.lock = threading.Lock()

        self.latest_message = None

        self.latest_objects = []

        self.last_update = 0


    def recv(
        self,
        frame
    ):

        imagen = (
            frame.to_ndarray(
                format="bgr24"
            )
        )

        alto_imagen, ancho_imagen = (
            imagen.shape[:2]
        )


        resultados = modelo.predict(
            source=imagen,
            conf=CONFIDENCE,
            imgsz=640,
            verbose=False
        )

        resultado = resultados[0]

        detecciones = []


        for caja in resultado.boxes:

            coords = (
                caja.xyxy[0]
                .cpu()
                .numpy()
            )

            x1, y1, x2, y2 = coords

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


            nombre = (
                TRADUCCION.get(
                    nombre_original,
                    nombre_original
                )
            )


            # =================================================
            # POSICIÓN
            # =================================================

            centro_x = (
                x1 + x2
            ) / 2

            posicion = (
                determinar_posicion(
                    centro_x,
                    ancho_imagen
                )
            )


            # =================================================
            # DISTANCIA
            # =================================================

            ancho_objeto_px = (
                x2 - x1
            )

            distancia = (
                calcular_distancia(
                    nombre_original,
                    ancho_objeto_px
                )
            )


            # =================================================
            # RIESGO
            # =================================================

            riesgo_base = (
                RIESGOS.get(
                    nombre_original,
                    "Bajo"
                )
            )

            riesgo = (
                ajustar_riesgo(
                    riesgo_base,
                    distancia
                )
            )


            objeto = {
                "nombre": nombre,
                "nombre_original": nombre_original,
                "confianza": confianza,
                "posicion": posicion,
                "distancia": distancia,
                "riesgo": riesgo
            }

            detecciones.append(
                objeto
            )


            # =================================================
            # DIBUJAR CAJA
            # =================================================

            color = (
                color_riesgo(
                    riesgo
                )
            )

            cv2.rectangle(
                imagen,
                (x1, y1),
                (x2, y2),
                color,
                3
            )


            if distancia is not None:

                etiqueta = (
                    f"{nombre} | "
                    f"{distancia:.1f} m"
                )

            else:

                etiqueta = (
                    nombre
                )


            cv2.putText(
                imagen,
                etiqueta,
                (
                    x1,
                    max(
                        30,
                        y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )


        # ====================================================
        # ACTUALIZAR MENSAJE
        # ====================================================

        tiempo_actual = (
            time.time()
        )

        if (
            tiempo_actual
            -
            self.last_update
            >=
            ANNOUNCE_EVERY
        ):

            mensaje = (
                construir_mensaje(
                    detecciones
                )
            )

            with self.lock:

                self.latest_message = (
                    mensaje
                )

                self.latest_objects = (
                    detecciones
                )

            self.last_update = (
                tiempo_actual
            )


        return (
            av.VideoFrame.from_ndarray(
                imagen,
                format="bgr24"
            )
        )


# ============================================================
# ESTILO ACCESIBLE
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #080808;
        color: white;
    }

    h1, h2, h3, p, label {
        color: white !important;
    }

    .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    button {
        min-height: 58px !important;
        font-size: 20px !important;
        font-weight: bold !important;
    }

    div[role="radiogroup"] label {
        background-color: #1a1a1a;
        border: 2px solid #444;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 10px;
        min-height: 60px;
        font-size: 22px !important;
        font-weight: bold !important;
        display: flex;
        align-items: center;
    }

    div[role="radiogroup"] label:hover {
        border-color: #1565C0;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# TÍTULO
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
        "📷 Reconocimiento"
    ]
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
        VisionMove utiliza la cámara del dispositivo
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
            "
        >
            🔊 Escuchar instrucciones
        </button>

        <script>

        function hablarInicio() {

            window.speechSynthesis.cancel();

            const texto =
                new SpeechSynthesisUtterance(
                    "Bienvenido a VisionMove. "
                    + "Este sistema reconoce objetos del entorno "
                    + "y proporciona información mediante voz. "
                    + "Para comenzar, seleccione Reconocimiento "
                    + "en el menú principal."
                );

            texto.lang = "es-CO";

            texto.rate = 0.9;

            texto.volume = 1.0;

            window.speechSynthesis.speak(
                texto
            );
        }

        </script>
        """,

        height=90
    )


    st.info(
        """
        VisionMove es un prototipo académico.
        No sustituye bastón blanco, perro guía
        ni técnicas profesionales de orientación
        y movilidad.
        """
    )


# ============================================================
# INSTRUCCIONES
# ============================================================

elif seccion == "📖 Instrucciones":

    st.header(
        "📖 Instrucciones de uso"
    )


    st.write(
        """
        **Paso 1.** Seleccione Reconocimiento.

        **Paso 2.** Pulse el botón START.

        **Paso 3.** Permita el acceso a la cámara.

        **Paso 4.** Oriente la cámara hacia el frente.

        **Paso 5.** VisionMove reconocerá objetos.

        **Paso 6.** Escuche la posición y distancia aproximada.
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
            "
        >
            🔊 Escuchar instrucciones completas
        </button>

        <script>

        function leerInstrucciones() {

            window.speechSynthesis.cancel();

            const texto =
                new SpeechSynthesisUtterance(
                    "Instrucciones de VisionMove. "
                    + "Paso uno. Seleccione Reconocimiento. "
                    + "Paso dos. Pulse el botón START. "
                    + "Paso tres. Permita el acceso a la cámara. "
                    + "Paso cuatro. Oriente la cámara hacia el frente. "
                    + "Paso cinco. VisionMove comenzará a reconocer objetos. "
                    + "El sistema indicará mediante voz "
                    + "si el objeto está a la izquierda, "
                    + "al frente o a la derecha, "
                    + "y su distancia aproximada."
                );

            texto.lang = "es-CO";

            texto.rate = 0.9;

            texto.volume = 1.0;

            window.speechSynthesis.speak(
                texto
            );
        }

        </script>
        """,

        height=90
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
        El siguiente control inicia la cámara.
        """
    )


    # ========================================================
    # BOTÓN DE AYUDA
    # ========================================================

    components.html(
        """
        <button
            onclick="indicarBoton()"
            style="
                width:100%;
                padding:20px;
                font-size:22px;
                background:#00897B;
                color:white;
                border:none;
                border-radius:14px;
            "
        >
            🔊 ¿Dónde inicio?
        </button>

        <script>

        function indicarBoton() {

            window.speechSynthesis.cancel();

            const texto =
                new SpeechSynthesisUtterance(
                    "El botón para comenzar se encuentra "
                    + "inmediatamente después de este mensaje. "
                    + "Deslice hacia abajo hasta encontrar START "
                    + "y púlselo. "
                    + "Luego permita el acceso a la cámara."
                );

            texto.lang = "es-CO";

            texto.rate = 0.9;

            texto.volume = 1.0;

            window.speechSynthesis.speak(
                texto
            );
        }

        </script>
        """,

        height=90
    )


    st.warning(
        """
        Mantenga el teléfono orientado hacia el frente
        durante el reconocimiento.
        """
    )


    # ========================================================
    # CÁMARA
    # ========================================================

     ctx = webrtc_streamer(

    key="visionmove-mobile",

    mode=WebRtcMode.SENDRECV,

    video_processor_factory=VisionMoveProcessor,

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
            }
        },
        "audio": False
    },

    async_processing=True
)

    # ========================================================
    # MEMORIA DE VOZctx
    # ========================================================

    if (
        "ultimo_mensaje_hablado"
        not in st.session_state
    ):

        st.session_state[
            "ultimo_mensaje_hablado"
        ] = ""


    # ========================================================
    # ACTUALIZAR RESULTADOS
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


        # ====================================================
        # MOSTRAR MENSAJE
        # ====================================================

        st.success(
            mensaje
        )


        # ====================================================
        # VOZ AUTOMÁTICA
        # ====================================================

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

                height=1
            )


            st.session_state[
                "ultimo_mensaje_hablado"
            ] = identificador


        # ====================================================
        # DETALLES
        # ====================================================

        if objetos:

            with st.expander(
                "Objetos detectados"
            ):

                for objeto in ordenar_detecciones(
                    objetos
                )[:5]:

                    if objeto[
                        "distancia"
                    ] is not None:

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

st.caption(
    """
    VisionMove • Prototipo académico
    de Ingeniería Biomédica
    """
)