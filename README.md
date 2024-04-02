# PDF2Chatbot (Streamlit-Local-Rag) : Chatbot de PDF con Streamlit y Langchain
![imagen](https://github.com/demiurg0/PDF2Chatbot/assets/165735354/ce9e90ce-69bc-4132-8c83-4f4e695c35e4)

Este proyecto implementa un chatbot interactivo capaz de responder preguntas basadas en el contenido de un archivo PDF. Utiliza Streamlit para la interfaz de usuario, LangChain para el procesamiento de lenguaje natural, y Ollama para las embeddings y consultas vectoriales.

## Características

- **Carga de archivos PDF**: Permite a los usuarios cargar documentos PDF desde los cuales el chatbot puede extraer información para responder preguntas.
- **Interfaz de chat interactiva**: Los usuarios pueden hacer preguntas y recibir respuestas en tiempo real a través de una interfaz de chat implementada con Streamlit.
- **Procesamiento avanzado de texto**: Utiliza LangChain para dividir el texto del PDF en fragmentos manejables y Ollama para generar embeddings de texto y realizar búsquedas semánticas.

## Requisitos

- Python 3.8+
- streamlit
- langchain
- loguru
- PyPDF2 o similar para la carga de PDFs
- Ollama

## Instalar Ollama y desplegar modelo ( llama2 en este caso)

Primero, clona el repositorio e instala las dependencias necesarias:

- Instalar Ollama, desplegar modelo llama2 ( o seleccionado ) -
  ```
  ollama run llama2
```
  
## Screenshot:
![Screenshot from 2024-04-01 18-01-09(1)](https://github.com/demiurg0/PDF2Chatbot/assets/165735354/fd5e6cf3-6aed-4c45-9fd7-d8a7470d452f)




# Documentación del Código

En este documento, proporcionaremos una descripción detallada del código proporcionado para un chatbot de PDF utilizando Streamlit y Langchain. El chatbot está diseñado para procesar archivos PDF, dividirlos en fragmentos manejables, y proporcionar respuestas a las preguntas de los usuarios sobre el contenido del PDF. A continuación, se explicará cada parte del código, desde las importaciones hasta la lógica de procesamiento de texto y la interacción con el usuario.

## Importaciones Necesarias

El código comienza importando las bibliotecas y módulos necesarios para su funcionamiento. Las importaciones incluyen componentes de Langchain, una biblioteca para procesamiento de lenguaje natural, así como Streamlit para la interfaz de usuario, y otras utilidades como Loguru para la gestión de registros.

```
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
```

## Configuración de Loguru y Directorios

El siguiente bloque de código configura Loguru para la gestión de registros y comprueba si existen los directorios necesarios para almacenar archivos generados por el chatbot.

```


logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")

if not os.path.exists('files'):
    os.mkdir('files')

if not os.path.exists('jj'):
    os.mkdir('jj')
```

## Configuración del Estado de la Sesión de Streamlit

Este bloque de código establece el estado inicial de la sesión de Streamlit, definiendo plantillas de mensajes y configuraciones de memoria para el historial de conversación.

```


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
```
## Interfaz de Usuario con Streamlit

El código utiliza Streamlit para crear una interfaz de usuario interactiva. Permite a los usuarios cargar un archivo PDF y ver el historial de chat. También incluye un cuadro de entrada para que los usuarios realicen preguntas.

```

st.title("PDF2Chatbot")

archivo_subido = st.file_uploader("Sube tu PDF", type='pdf')

# Mostrar historial de chat
for mensaje in st.session_state.chat_history:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["message"])

# Procesamiento del archivo PDF subido
if archivo_subido is not None:
    # Lógica de procesamiento del PDF y chat
    ...
else:
    st.write("Por favor, sube un archivo PDF.")
```
## Lógica de Procesamiento del PDF y Chat

El código procesa el archivo PDF subido dividiéndolo en fragmentos manejables, crea y persiste una tienda de vectores para el contenido del PDF, y luego utiliza un modelo de preguntas y respuestas para interactuar con el usuario.

```

# Lógica de Procesamiento del PDF y Chat
if archivo_subido is not None:
    # Lógica de procesamiento del PDF y chat
    ...
else:
    st.write("Por favor, sube un archivo PDF.")
```
## Evaluación y Ajuste de Parámetros

Se evalúa la longitud del texto del PDF para ajustar los parámetros de división de texto según sea necesario.

```

# Evaluación y Ajuste de Parámetros
def evaluar_y_ajustar_parametros(longitud_del_texto):
    # Código de evaluación y ajuste de parámetros
    ...
```
