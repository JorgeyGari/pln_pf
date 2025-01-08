# Sistema de verificación de hechos

Repositorio contenedor del código y la documentación del sistema de verificación de hechos desarrollado para la práctica final del curso Procesamiento de Lenguaje Natural, impartido por la Universidad Carlos III de Madrid en el Máster Universitario de Inteligencia Artificial Aplicada durante el año 2024-2025. El sistema se ha implementado utilizando RAG (Retrieval-Augmentated Generation).

## Descripción

El sistema de verificación de hechos permite la extracción de información de una base de datos, en este caso, Wikipedia, y con esa información se determina la veracidad de una pregunta o afirmación. El sistema es capaz de razonar la respuesta y minimizar las alucinaciones siempre que cuente con información relevante y adecuada.

## Características

- **Extracción de información**: Utiliza Wikipedia como base de datos para obtener información relevante.
- **Verificación de hechos**: Determina la veracidad de preguntas o afirmaciones basándose en la información extraída.
- **Razonamiento**: Capacidad de razonar las respuestas proporcionadas mediante varias cadenas secuenciadas.
- **Minimización de alucinaciones**: Reduce la generación de respuestas incorrectas utilizando razonamiento y recuperación de información.

## Requisitos

- Python 3.11 o superior
- Bibliotecas necesarias (ver [`requirements.txt`](requirements.txt))

## Instalación

1. Clona este repositorio:
    ```bash
    git clone https://github.com/JorgeyGari/pln_pf.git
    ```
2. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

1. Ejecuta el script principal:
    ```bash
    python src/fact_checker.py
    ```
2. Introduce una pregunta o afirmación en la consola para verificar su veracidad.

## Desarrolladores

* Alejandro Climent
* Jorge Lázaro
* Aimar Nicuesa
* Daniel Obreo
