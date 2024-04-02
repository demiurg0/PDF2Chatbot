# Importaciones necesarias
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from loguru import logger
from typing import List, Optional
import streamlit as st
import os
import time
import sys
sys.modules['sqlite3'] = __import__('pysqlite3')

# Configuración de loguru para el registro de información
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")

# Crear directorios si no existen
if not os.path.exists('files'):
    os.mkdir('files')

if not os.path.exists('jj'):
    os.mkdir('jj')

# Configurar estado de la sesión de Streamlit
def init_session_state():
    if 'template' not in st.session_state:
        st.session_state.template = """Eres un chatbot conocedor, aquí para ayudar con las preguntas del usuario. Tu tono debe ser profesional e informativo.

        Contexto: {context}
        Historial: {history}

        Usuario: {question}
        Chatbot:"""
    if 'prompt' not in st.session_state:
        st.session_state.prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=st.session_state.template,
        )
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            input_key="question")
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = Chroma(persist_directory='jj',
                                              embedding_function=OllamaEmbeddings(base_url='http://localhost:11434',
                                                                                  model="llama2")
                                              )
    if 'llm' not in st.session_state:
        st.session_state.llm = Ollama(base_url="http://localhost:11434",
                                      model="llama2",
                                      verbose=True,
                                      callback_manager=CallbackManager(
                                          [StreamingStdOutCallbackHandler()]),
                                      )

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

init_session_state()

st.title("PDF2Chatbot")

def evaluar_y_ajustar_parametros(longitud_del_texto):
    if longitud_del_texto > 50000:  # Si el texto es muy largo
        return 4000, 500  # tamaño de fragmento grande con más solapamiento
    elif longitud_del_texto > 20000:  # Si el texto es de longitud media
        return 3000, 400
    else:  # Si el texto es corto
        return 2000, 300

archivo_subido = st.file_uploader("Sube tu PDF", type='pdf')


# Mostrar historial de chat
for mensaje in st.session_state.chat_history:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["message"])

# Procesamiento del archivo PDF subido
if archivo_subido is not None:
    ruta_del_archivo: str = f"files/{archivo_subido.name}"
    if not os.path.isfile(ruta_del_archivo):
        with st.spinner("Analizando tu documento..."):
            datos_en_bytes: bytes = archivo_subido.read()
            with open(ruta_del_archivo, "wb") as f:
                f.write(datos_en_bytes)
                logger.info(f"Archivo PDF guardado en {ruta_del_archivo}")

            cargador = PyPDFLoader(ruta_del_archivo)
            texto_del_pdf: str = cargador.load()
            logger.info("Contenido del PDF cargado y listo para el procesamiento")

            # Ajustar parámetros basados en la evaluación del PDF
            tamaño_del_fragmento, solapamiento = evaluar_y_ajustar_parametros(len(texto_del_pdf))

            # Inicializar el divisor de texto con parámetros ajustados
            divisor_de_texto: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
                chunk_size=tamaño_del_fragmento,
                chunk_overlap=solapamiento,
                length_function=len
            )
            todos_los_fragmentos: List[str] = divisor_de_texto.split_documents(texto_del_pdf)
            logger.info("Texto dividido en fragmentos manejables para el procesamiento de LLM")

            # Crear y persistir la tienda de vectores
            tienda_de_vectores: Chroma = Chroma.from_documents(
                documents=todos_los_fragmentos,
                embedding=OllamaEmbeddings(base_url='http://localhost:11434', model="llama2"),
                persist_directory='jj'
            )
            tienda_de_vectores.persist()
            logger.info("Tienda de vectores creada y persistida")
            st.session_state.vectorstore = tienda_de_vectores

    st.session_state.retriever = st.session_state.vectorstore.as_retriever()

    # Inicialización de la cadena de preguntas y respuestas
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

    # Input de chat
    if entrada_del_usuario := st.chat_input("Tú:", key="entrada_del_usuario"):
        mensaje_del_usuario = {"role": "user", "message": entrada_del_usuario}
        st.session_state.chat_history.append(mensaje_del_usuario)
        with st.chat_message("user"):
            st.markdown(entrada_del_usuario)
        with st.chat_message("assistant"):
            with st.spinner("El asistente está escribiendo..."):
                respuesta = st.session_state.qa_chain(entrada_del_usuario)
            placeholder_del_mensaje = st.empty()
            respuesta_completa = ""
            for fragmento in respuesta['result'].split():
                respuesta_completa += fragmento + " "
                time.sleep(0.05)
                # Añadir un cursor parpadeante para simular la escritura
                placeholder_del_mensaje.markdown(respuesta_completa + "▌")
            placeholder_del_mensaje.markdown(respuesta_completa)

        mensaje_del_chatbot = {"role": "assistant", "message": respuesta['result']}
        st.session_state.chat_history.append(mensaje_del_chatbot)

else:
    st.write("Por favor, sube un archivo PDF.")
