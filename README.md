# Sistema de Verificación de Hechos

Este repositorio contiene el código y la documentación del sistema de verificación de hechos desarrollado en esta práctica. El sistema se ha implementado utilizando un enfoque de Recuperación-Augmentación-Generación (RAG).

## Descripción

El sistema de verificación de hechos permite la extracción de información de una base de datos, en este caso, Wikipedia, y con esa información se determina la veracidad de una pregunta o afirmación. El sistema es capaz de razonar la respuesta y minimizar las alucinaciones, aunque puede generar respuestas incorrectas si no dispone del contexto adecuado.

## Características

- **Extracción de información**: Utiliza Wikipedia como base de datos para obtener información relevante.
- **Verificación de hechos**: Determina la veracidad de preguntas o afirmaciones basándose en la información extraída.
- **Razonamiento**: Capacidad de razonar las respuestas proporcionadas.
- **Minimización de alucinaciones**: Reduce la generación de respuestas incorrectas, aunque no las elimina por completo.

## Requisitos

- Python 3.x
- Bibliotecas necesarias (ver `requirements.txt`)

## Instalación

1. Clona este repositorio:
    ```bash
    git clone https://github.com/JorgeyGari/pln_pf.git
    ```
2. Navega al directorio del proyecto:
    ```bash
    cd pln_pf/src
    ```
3. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

1. Ejecuta el script principal:
    ```bash
    python fact_checker.py
    ```
2. Introduce una pregunta o afirmación para verificar su veracidad.

## Contacto

Para cualquier consulta, puedes contactarme en jorgelazaro2002@hotmail.com.
