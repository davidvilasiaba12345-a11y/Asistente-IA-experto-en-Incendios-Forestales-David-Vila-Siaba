import os
import time
import warnings

import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver



"""
================================================================================
CONFIGURACIÓN INICIAL
================================================================================

"""

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

MODEL = "gemini-3.5-flash-lite"
MODEL_EMBEDDING = "gemini-embedding-2"
RUTA_CHROMA = "data/chroma_incendios"
NOMBRE_COLECCION = "incendios_forestales"
K_RECUPERACION = 5

NOMBRES_FUENTES = {
    "incendios-decenio-2006-2015_tcm30-521617.pdf": "Los Incendios Forestales en España. Decenio 2006-2015 (Ministerio de Agricultura, Pesca y Alimentación)",
    "Estadística-General-IF-Metodología.pdf": "Metodología de la Estadística General de Incendios Forestales (MITECO)",
    "Consejos_Incendios_Forestales.pdf": "Consejos ante incendios forestales (Protección Civil, Ayuntamiento de Jerez)",
    "fires-all.csv": "Estadística General de Incendios Forestales, registros 1968-2022"
}

PREGUNTAS_EJEMPLO = [
    "¿Cuál fue el peor año de incendios forestales en España?",
    "¿Por qué se provocan los incendios intencionados?",
    "¿Qué ocurrió en el incendio de Minas de Riotinto?",
    "¿Qué hago si me sorprende un incendio en el monte?"
]

st.set_page_config(
    page_title="Vigía Forestal",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

"""
================================================================================
PROMPT Y CONTEXTO
================================================================================

"""

def formatear_contexto(documentos: list) -> str:
    """Convierte los documentos recuperados en un texto con su fuente identificada"""
    if not documentos:
        return "No se ha recuperado ningún documento."

    bloques = []
    for i, doc in enumerate(documentos, start=1):
        fuente = NOMBRES_FUENTES.get(doc.metadata["fuente"], doc.metadata["fuente"])  # Si no recomozco el archivo, uso su nombre tal cual
        if doc.metadata["tipo"] == "pdf":
            fuente += f", página {doc.metadata['pagina']}"
        bloques.append(f"[Documento {i} | Fuente: {fuente}]\n{doc.page_content}")

    return "\n\n---\n\n".join(bloques)

def construir_system_prompt(contexto: str, pregunta_interpretada: str = "") -> str:
    """Genera el system prompt del asistente insertando el contexto recuperado."""
    bloque_pregunta = (
        f"\n## PREGUNTA ACTUAL\n"
        f"Teniendo en cuenta el historial de la conversación, la última pregunta del usuario se interpreta como:\n"
        f"\"{pregunta_interpretada}\"\n"
        f"Responde a esta pregunta interpretada.\n"
    ) if pregunta_interpretada else ""

    return f"""
        ## ROL
        Eres un asistente experto en incendios forestales en España. Actúas como un analista especializado en estadística forestal y protección civil que, además, sabe explicar la información técnica de forma clara a cualquier persona, sin necesidad de conocimientos previos.

        ## OBJETIVO
        Responder a las preguntas del usuario sobre incendios forestales en España (estadísticas, causas, grandes incendios, evolución histórica, distribución territorial, prevención y autoprotección) utilizando EXCLUSIVAMENTE la información de los documentos que se te proporcionan en la sección CONTEXTO.

        ## FUENTES DE TU BASE DE CONOCIMIENTO
        - Informe "Los Incendios Forestales en España. Decenio 2006-2015": análisis de causas, grandes incendios, distribución territorial y temporal, extinción y prevención.
        - Metodología de la Estadística General de Incendios Forestales: definiciones de las variables y categorías de causa.
        - Consejos de Protección Civil: recomendaciones de prevención y autoprotección.
        - Estadística General de Incendios Forestales 1968-2022: resúmenes por año, por comunidad autónoma y fichas de los incendios de 3.000 hectáreas o más.

        ## INSTRUCCIONES
        1. Basa tu respuesta únicamente en el CONTEXTO. No utilices tu conocimiento general para aportar datos, cifras, fechas ni nombres que no aparezcan en él.
        2. Antes de responder, comprueba si el CONTEXTO contiene la información necesaria:
        - Si la contiene, responde con ella.
        - Si solo la contiene en parte, responde a lo que puedas e indica claramente qué parte no está disponible.
        - Si no la contiene, responde: "No dispongo de información sobre eso en mi base de conocimiento." y sugiere brevemente qué tipo de preguntas sí puedes responder.
        3. Reproduce las cifras exactamente como aparecen en el CONTEXTO, con sus unidades (hectáreas, %, número de incendios) y su periodo de referencia. Si dos fuentes ofrecen datos de periodos distintos (por ejemplo, 2006-2015 frente a 1968-2022), no los mezcles: indica a qué periodo corresponde cada uno.
        4. Ten en cuenta el historial de la conversación para interpretar preguntas de seguimiento (por ejemplo, "¿y en Galicia?" o "¿cuántas hectáreas quemó ese incendio?"), pero responde siempre con la información del CONTEXTO.
        5. Responde solo sobre incendios forestales en España. Si el usuario pregunta por otro tema, indícale amablemente que solo puedes ayudarle con este ámbito. Puedes responder con normalidad a saludos y agradecimientos.
        6. Si el usuario describe una emergencia o un incendio en curso, indícale en primer lugar que llame al 112 y, a continuación, ofrécele las pautas de autoprotección que aparezcan en el CONTEXTO.
        7. Nunca proporciones información que pueda servir para provocar un incendio o dificultar su extinción.
        8. El CONTEXTO puede incluir documentos que no responden a la pregunta (por ejemplo, de otro territorio o de otro periodo). Utiliza solo los documentos que se refieran a lo que pregunta el usuario e ignora el resto, sin mencionarlos en tu respuesta.

        ## FORMATO DE RESPUESTA
        - Responde siempre en español, con un tono claro, profesional y cercano.
        - Dirígete siempre al usuario de tú, también cuando reproduzcas recomendaciones de los documentos que estén redactadas de usted.
        - Empieza por la respuesta directa a la pregunta y amplía después con los detalles relevantes.
        - Sé conciso: no superes las 200 palabras salvo que el usuario pida más detalle.
        - Usa listas cuando enumeres varios elementos (causas, consejos, años, comunidades).
        - Termina con una línea "📚 Fuentes:" seguida de las fuentes que hayas utilizado realmente para construir la respuesta. Cítalas con el texto que aparece tras "Fuente:" en la etiqueta de cada documento, sin el número de documento. Si una misma fuente aparece en varios documentos, cítala una sola vez (en el caso de los PDF, agrupando sus páginas, por ejemplo "páginas 61 y 67"). Si no has utilizado información del CONTEXTO, porque no dispones de ella o porque la pregunta está fuera de tu ámbito, no incluyas esta línea.
        {bloque_pregunta}
        ## CONTEXTO
        <contexto>
        {contexto}
        </contexto>
    """

"""
================================================================================
CREACCIÓN DEL AGENTE
================================================================================

"""

class EstadoAgente(MessagesState):
    pregunta_busqueda: str  # Pregunta reformulada que se usa para buscar en ChromaDB
    contexto: str           # Documentos recuperados, ya formateados para el prompt

@st.cache_resource
def crear_agente():
    """Carga la base vectorial y compila el agente de LangGraph con memoria"""
    llm = ChatGoogleGenerativeAI(model=MODEL, google_api_key=API_KEY)
    embeddings = GoogleGenerativeAIEmbeddings(model=MODEL_EMBEDDING, api_key=API_KEY)

    vectorstore = Chroma(
        persist_directory=RUTA_CHROMA,
        embedding_function=embeddings,
        collection_name=NOMBRE_COLECCION
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": K_RECUPERACION})

    def nodo_reformular(estado: EstadoAgente) -> dict:
        """Reescribe la última pregunta para que se entienda sin el historial de la conversación"""
        pregunta = estado["messages"][-1].text
        preguntas_previas = [m for m in estado["messages"][:-1] if m.__class__.__name__ == "HumanMessage"]

        if not preguntas_previas:
            return {"pregunta_busqueda": pregunta}

        historial = "\n".join(
            f"{'Usuario' if m.__class__.__name__ == 'HumanMessage' else 'Asistente'}: {m.text[:500]}"
            for m in estado["messages"][-7:-1]  # Uso los 6 mensajes anteriores a la pregunta actual
        )

        prompt = f"""Tu tarea es reescribir la última pregunta del usuario para que se entienda por sí sola, sin necesidad de leer la conversación anterior.
                Sustituye las referencias como "ese incendio", "ese año", "allí" o "¿y en Galicia?" por los nombres, fechas o lugares concretos que aparecen en el historial.
                Si la pregunta ya se entiende por sí sola, devuélvela sin cambios.
                No respondas a la pregunta. Devuelve ÚNICAMENTE la pregunta reescrita.

                <historial>
                {historial}
                </historial>

                <pregunta>
                {pregunta}
                </pregunta>"""

        return {"pregunta_busqueda": llm.invoke(prompt).text.strip()}

    def nodo_recuperar(estado: EstadoAgente) -> dict:
        """Recupera de ChromaDB los documentos más relevantes para la pregunta reformulada"""
        documentos = retriever.invoke(estado["pregunta_busqueda"])
        return {"contexto": formatear_contexto(documentos)}

    def nodo_generar(estado: EstadoAgente) -> dict:
        """Genera la respuesta con Gemini a partir del system prompt, el contexto y el historial"""
        mensajes = [SystemMessage(content=construir_system_prompt(estado["contexto"], estado["pregunta_busqueda"]))] + estado["messages"]
        return {"messages": [llm.invoke(mensajes)]}

    grafo = StateGraph(EstadoAgente)

    grafo.add_node("reformular", nodo_reformular)
    grafo.add_node("recuperar", nodo_recuperar)
    grafo.add_node("generar", nodo_generar)

    grafo.add_edge(START, "reformular")
    grafo.add_edge("reformular", "recuperar")
    grafo.add_edge("recuperar", "generar")
    grafo.add_edge("generar", END)

    return grafo.compile(checkpointer=InMemorySaver())

"""
================================================================================
ESTADO DE LA SESIÓN
================================================================================

"""

def nueva_conversacion():
    """Reinicia el historial visible y asigna un nuevo identificador de conversación"""
    st.session_state.messages = []
    st.session_state.thread_id = f"streamlit-{time.time()}"

if "messages" not in st.session_state:
    nueva_conversacion()

"""
================================================================================
COMPROBACIONES PREVIAS
================================================================================

"""

if not API_KEY:
    st.error("❌ No se ha encontrado la API key de Gemini. Configura la variable GEMINI_API_KEY en el archivo .env o en los secrets de Streamlit.")
    st.stop()

if not os.path.exists(RUTA_CHROMA):
    st.error("❌ No se ha encontado la base vectorial en 'data/chroma_incendios'. Ejecuta primero el notebook para generarla.")
    st.stop()

agente = crear_agente()

"""
================================================================================
SIDEBAR
================================================================================

"""

with st.sidebar:
    st.title("🔥 Vigía Forestal")
    st.caption("Asistente experto en incendios forestales en España con Gemini, RAG y LangGraph")
    st.divider()

    st.subheader("📚 Base de conocimiento")
    st.markdown(
        "- Informe *Los Incendios Forestales en España. Decenio 2006-2015*\n"
        "- Metodología de la Estadística General de Incendios Forestales\n"
        "- Consejos de autoprotección de Protección Civil\n"
        "- Estadística de incendios forestales 1968-2022"
    )
    st.divider()

    st.subheader("💡 Preguntas de ejemplo")
    for ejemplo in PREGUNTAS_EJEMPLO:
        if st.button(ejemplo, use_container_width=True):
            st.session_state.pregunta_pendiente = ejemplo   # La guardo para enviarla en el bloque del chat

    st.divider()

    if st.button("🆕 Nueva conversación", use_container_width=True, type="primary"):
        nueva_conversacion()
        st.rerun()

    st.caption("⚠️ Ante cualquier incendio, llama al **112**.")