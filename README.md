# Streamlit-Local-Rag : Chatbot de PDF con Streamlit y Langchain

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

## Instalación

Primero, clona el repositorio e instala las dependencias necesarias:


