# Importaciones estándar
import sys
import os
import shutil
import uuid
import tempfile
import logging
from typing import Any, Dict, Optional, List

# Importaciones de terceros
import streamlit as st
from decouple import config
import together
from PyPDF2 import PdfReader
from pydantic import BaseModel, root_validator

# Importaciones para la implementación con Langchain y Together
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# Importaciones de langchain_community para Ollama
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings

# Loguru para registros
from loguru import logger

# Corrección específica para despliegue en Streamlit
sys.modules['sqlite3'] = __import__('pysqlite3')

# Configuración de loguru para el registro de información
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")

# Configuración de la página en Streamlit
st.set_page_config(page_title='Pregunta a tu PDF', 
                   page_icon="📄", 
                   layout="wide")

# Clase para LLM de Together
class TogetherLLM(LLM):
    model = "togethercomputer/llama-2-70b-chat"
    together_api_key = config("TOGETHER_API_KEY")
    temperature = 0.1
    max_tokens = 1024

    class Config:
        extra = 'forbid'

    @root_validator()
    def validate_environment(cls, values):
        api_key = values.get("together_api_key") or os.getenv("TOGETHER_API_KEY")
        values["together_api_key"] = api_key
        return values
    
    @property
    def _llm_type(self):
        return "together"

    def _call(self, prompt, **kwargs):
        together.api_key = self.together_api_key
        output = together.Complete.create(
            prompt,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        text = output['output']['choices'][0]['text']
        return text

# Interfaz de usuario para la selección del modelo
st.sidebar.title("Configuración")
model_selection = st.sidebar.selectbox(
    "Elige el modelo de LLM",
    ("TogetherLLM (Llama 2)", "Ollama")
)

# Inicialización del LLM según la selección del usuario
if model_selection == "TogetherLLM (Llama 2)":
    llm = TogetherLLM()
    embedding_model_name = "BAAI/bge-base-en"
else:  # Ollama seleccionado
    llm = Ollama(base_url="http://localhost:11434", model="llama2", verbose=True)
    embedding_model_name = OllamaEmbeddings(base_url='http://localhost:11434', model="llama2")

# Proceso general de carga y análisis de PDF
uploaded_file = st.file_uploader('Sube un archivo', type='pdf')
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

# Continuación del manejo del archivo PDF subido
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

    # Procesamiento del PDF: Cargar y extraer texto
    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        logger.info("PDF cargado y procesado exitosamente.")

        # Dividir el texto en fragmentos para su procesamiento
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        logger.info(f"Documento dividido en {len(docs)} fragmentos.")

        # Preparación de embeddings y Chroma DB según la selección del modelo
        if model_selection == "TogetherLLM (Llama 2)":
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        else:  # Ollama seleccionado
            embeddings = OllamaEmbeddings(base_url='http://localhost:11434', model="llama2")

        # Creación de la base de datos Chroma
        session_uuid = uuid.uuid4()
        db_name = f"./data/chroma_db_{session_uuid}.db"
        db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=db_name)
        logger.info("Base de datos Chroma creada y persistida.")

        # Interfaz de usuario para realizar consultas al modelo
        query_text = st.text_input("Escribe tu pregunta aquí:")
        if query_text:
            # Creación de la cadena de preguntas y respuestas (QA)
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"return_source_documents": True},
            )

            # Realización de la consulta
            response = qa(query_text)
            st.write("Respuesta:", response['result'])
    except Exception as e:
        logger.error(f"Error al procesar el archivo PDF: {e}")
        st.error("Hubo un error al procesar tu PDF. Por favor, intenta con otro archivo.")

# Limpieza: Código para manejar la eliminación de archivos temporales y bases de datos si es necesario
# Es importante para evitar la acumulación de archivos no deseados en el servidor
finally:
    if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
        os.remove(temp_file_path)
    if 'db_name' in locals() and os.path.exists(db_name):
        shutil.rmtree(os.path.dirname(db_name))

