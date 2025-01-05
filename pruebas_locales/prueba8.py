import os
os.environ['OLLAMA_HOST'] = 'http://localhost:11434'
import wikipedia
import wikipediaapi
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import ollama
import re
from itertools import islice
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

wikipedia.set_lang("es")
wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)','es')

def extraer_ngrams(statement, n=2):
    palabras = re.findall(r'\b\w+\b', statement.lower())
    stop_words = {'el', 'la', 'los', 'las', 'de', 'y', 'en', 'a', 'que', 'un', 'una', 'con', 'por', 'para', 'es', 'al', 'del', 'se', 'su', 'lo'}
    palabras_filtradas = [palabra for palabra in palabras if palabra not in stop_words]
    ngrams = zip(*(islice(palabras_filtradas, i, None) for i in range(n)))
    frases = [" ".join(ngram) for ngram in ngrams]
    return frases

def buscar_en_wikipedia(statement):
    frases = extraer_ngrams(statement, n=2)
    busqueda = " ".join(frases)

    titulos = wikipedia.search(busqueda, results=10)

    if not titulos:
        return None, []

    paginas_contenido = []

    for titulo in titulos:
        page = wiki_wiki.page(titulo)
        if page.exists():
            texto = page.text
            paginas_contenido.append((titulo, texto))

    # Ordenar páginas por la cantidad de contenido
    paginas_contenido.sort(key=lambda x: len(x[1]), reverse=True)

    # Seleccionar las 3 páginas con más contenido
    paginas_seleccionadas = paginas_contenido[:5]

    textos_recuperados = [pagina[1] for pagina in paginas_seleccionadas]
    fuentes = [pagina[0] for pagina in paginas_seleccionadas]

    return textos_recuperados, fuentes

def evaluate_truthfulness(statement):
    retrieved_docs, fuentes = buscar_en_wikipedia(statement)
    if not retrieved_docs:
        retrieved_docs = ["No se encontró información relevante en Wikipedia."]
        fuentes = ["Sin fuentes"]

    contexto = "\n\n".join([f"Contenido de {fuente}:\n{texto}" for fuente, texto in zip(fuentes, retrieved_docs)])
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                'role': 'user',
                'content': f"Contexto:\n{contexto}\n\nPregunta: ¿Es verdadera la siguiente declaración?\nDeclaración: {statement}\nExplica por qué basándote solo en el contexto proporcionado. Si no hay suficiente información, intenta estimar con base en el contexto, pero justifica tu respuesta:"
            },
        ]
    )

    generated_response = response['message']['content']

    # Segundo paso: Solicitar una respuesta categórica
    classification_response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                'role': 'user',
                'content': f"Con base en el análisis anterior:\n{generated_response}\n\nResponde únicamente con una palabra: 'Verdadera' o 'Falsa'."
            },
        ]
    )

    classification = classification_response['message']['content'].strip().lower()
    print(classification)
    is_truthful = classification == "verdadera"
        
    return {
        "statement": statement,
        "response": generated_response,
        "is_truthful": is_truthful,
        "sources": fuentes
    }

statement = input("Por favor, ingresa una afirmación para verificar su veracidad: ")
result = evaluate_truthfulness(statement)

print(f"Declaración: {result['statement']}")
print("Veracidad: Verdadera" if result['is_truthful'] else "Veracidad: Falsa")
print(f"Justificación: {result['response']}")
print(f"Fuentes utilizadas: {', '.join(set(result['sources']))}")
