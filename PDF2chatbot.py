import os
import sys
import threading
from typing import Dict, List, Optional
from langchain.chains import RetrievalQA
#from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from loguru import logger
import streamlit as st

sys.modules['sqlite3'] = __import__('pysqlite3')

# Configuración inicial
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
if not os.path.exists('files'):
    os.mkdir('files')
if not os.path.exists('jj'):
    os.mkdir('jj')

# Estado de procesamiento global
estado_procesamiento = {
    "listo": False,
    "mensaje": "Inicializando...",
    "tiempo_estimado": "Calculando...",
    "debug": []
}

def actualizar_estado(mensaje: str, tiempo_estimado: Optional[str] = None, debug: Optional[str] = None):
    estado_procesamiento["mensaje"] = mensaje
    if tiempo_estimado is not None:
        estado_procesamiento["tiempo_estimado"] = tiempo_estimado
    if debug is not None:
        estado_procesamiento["debug"].append(debug)
    # Note: Esta función podría requerir mecanismos específicos para asegurar la actualización segura entre hilos.

def evaluar_y_ajustar_parametros(longitud_del_texto: int) -> (int, int):
    if longitud_del_texto > 50000:
        return 4000, 500
    elif longitud_del_texto > 20000:
        return 3000, 400
    else:
        return 2000, 300

cache_de_tienda_de_vectores: Dict[str, Chroma] = {}

def obtener_o_crear_tienda_de_vectores(documentos: List[str], clave_del_documento: str) -> Chroma:
    if clave_del_documento in cache_de_tienda_de_vectores:
        logger.info("Recuperando tienda de vectores desde el caché")
        return cache_de_tienda_de_vectores[clave_del_documento]
    else:
        logger.info("Creando nueva tienda de vectores y añadiéndola al caché")
        tienda_de_vectores = Chroma.from_documents(
            documents=documentos,
            embedding=OllamaEmbeddings(base_url='http://localhost:11434', model="llama2"),
            persist_directory='jj'
        )
        tienda_de_vectores.persist()
        cache_de_tienda_de_vectores[clave_del_documento] = tienda_de_vectores
        return tienda_de_vectores

def procesar_pdf_async(ruta_del_archivo: str, callback):
    try:
        actualizar_estado("Cargando PDF...", debug=f"Cargando {ruta_del_archivo}")
        cargador = PyPDFLoader(ruta_del_archivo)
        texto_del_pdf = cargador.load()
        actualizar_estado("PDF cargado. Procesando...", "Unos segundos más", debug="PDF cargado")

        tamaño_del_fragmento, solapamiento = evaluar_y_ajustar_parametros(len(texto_del_pdf))
        divisor_de_texto = RecursiveCharacterTextSplitter(chunk_size=tamaño_del_fragmento, chunk_overlap=solapamiento, length_function=len)
        todos_los_fragmentos = divisor_de_texto.split_documents(texto_del_pdf)
        
        tienda_de_vectores = obtener_o_crear_tienda_de_vectores(todos_los_fragmentos, ruta_del_archivo)

        actualizar_estado("Procesamiento completado. Inicializando chat...", "Casi listo", debug="Tienda de vectores lista")
        callback(tienda_de_vectores)
    except Exception as e:
        actualizar_estado("Error en el procesamiento", debug=str(e))
        callback(None)

def cargar_y_procesar_pdf(archivo_subido):
    ruta_del_archivo = f"files/{archivo_subido.name}"
    with open(ruta_del_archivo, "wb") as f:
        f.write(archivo_subido.getbuffer())
    actualizar_estado("Archivo subido. Comenzando procesamiento...", debug=f"Archivo {archivo_subido.name} guardado.")
    threading.Thread(target=procesar_pdf_async, args=(ruta_del_archivo, on_pdf_processed)).start()

def on_pdf_processed(tienda_de_vectores):
    if tienda_de_vectores:
        estado_procesamiento["listo"] = True
        actualizar_estado("Sistema de chat listo para usar.", "Listo", debug="Inicialización completa")
        st.session_state.tienda_de_vectores = tienda_de_vectores
        init_qa_chain(tienda_de_vectores)
    else:
        st.error("Hubo un error al procesar el archivo PDF.")

def init_qa_chain(tienda_de_vectores):
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=Ollama(base_url="http://localhost:11434", model="llama2"),
        chain_type='retrieval',
        retriever=tienda_de_vectores.as_retriever(),
        verbose=True
    )

def init_session_state():
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def mostrar_estado_procesamiento():
    st.write(estado_procesamiento["mensaje"])
    if estado_procesamiento["tiempo_estimado"] != "Calculando...":
        st.write(f"Tiempo estimado restante: {estado_procesamiento['tiempo_estimado']}")
    for debug_msg in estado_procesamiento["debug"][-5:]:  # Mostrar los últimos 5 mensajes de depuración
        st.text(debug_msg)

def manejar_entrada_chat(entrada_del_usuario: str):
    if st.session_state.qa_chain is not None:
        with st.spinner('El asistente está escribiendo...'):
            respuesta = st.session_state.qa_chain(entrada_del_usuario)
        st.session_state.chat_history.append({"role": "user", "message": entrada_del_usuario})
        st.session_state.chat_history.append({"role": "assistant", "message": respuesta['result']})
    else:
        st.error("El sistema de chat aún no está listo. Por favor, espera.")

init_session_state()
st.title("Chatbot de PDF Optimizado")

archivo_subido = st.file_uploader("Sube tu PDF", type='pdf')
if archivo_subido:
    cargar_y_procesar_pdf(archivo_subido)

mostrar_estado_procesamiento()

entrada_del_usuario = st.text_input("Escribe tu pregunta:", key="entrada_del_usuario")
if entrada_del_usuario:
    manejar_entrada_chat(entrada_del_usuario)
