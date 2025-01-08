"""
Este script descarga el texto de páginas de Wikipedia en español y guarda el contenido en archivos de texto.
Funciones:
    get_wikipedia_text(title):
        Descarga el texto de una página de Wikipedia dado su título.
        Parámetros:
            title (str): El título de la página de Wikipedia.
        Retorna:
            str: El texto de la página de Wikipedia si existe, de lo contrario, un mensaje indicando que la página no existe.
Variables:
    titulos (list): Lista de títulos de las páginas de Wikipedia que se desean obtener.
    respuesta (str): Entrada del usuario para el título de la página de Wikipedia.
Proceso:
    1. Solicita al usuario que introduzca los títulos de las páginas de Wikipedia que desea obtener.
    2. Para cada título introducido, descarga el texto de la página de Wikipedia.
    3. Guarda el texto descargado en un archivo de texto con el nombre del título de la página.
    4. Informa al usuario cuando el proceso ha sido completado.
"""

import wikipediaapi

def get_wikipedia_text(title):
    wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)','es')
    page = wiki_wiki.page(title)
    
    if page.exists():
        print(f"Descargando texto de la página '{title}'...")
        return page.text
    else:
        print(f"La página '{title}' no existe.")
        return "La página no existe."

# Títulos de las páginas de Wikipedia que quieres obtener
titulos = []
respuesta = input("Introduce el título de la página de Wikipedia que quieres obtener (o 'fin' para terminar): ")
while respuesta != "fin":
    titulos.append(respuesta)
    respuesta = input("Introduce el título de la página de Wikipedia que quieres obtener (o 'fin' para terminar): ")

for title in titulos:
    text = get_wikipedia_text(title)
    with open(f"{title}.txt", "w", encoding="utf-8") as file:
        file.write(text)
    print(f"Texto de la página '{title}' guardado en {title}.txt")

print("Proceso completado.")