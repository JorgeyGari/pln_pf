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

def dividir_en_frases(texto):
    frases = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)\s(?=\w)', texto)
    return [frase.strip() for frase in frases if frase.strip()]

def filtrar_titulos(statement, titulos):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([statement] + titulos)
    similitudes = cosine_similarity(X[0:1], X[1:])
    umbral = 0.2
    titulos_relevantes = [titulo for i, titulo in enumerate(titulos) if similitudes[0][i] > umbral]
    return titulos_relevantes

def buscar_en_wikipedia(statement):
    frases = extraer_ngrams(statement, n=2)
    busqueda = " ".join(frases)

    titulos = wikipedia.search(busqueda, results=10)
    titulos = filtrar_titulos(statement, titulos)

    if not titulos:
        return None, []

    textos_recuperados = []
    fuentes = []

    for titulo in titulos:
        page = wiki_wiki.page(titulo)
        if page.exists():
            texto = page.text
            secciones = texto.split('\n\n')
            textos_recuperados.extend(secciones)
            fuentes.extend([titulo] * len(secciones))

    return textos_recuperados, fuentes

def filtrar_contenido(frases, statement):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([statement] + frases)
    query_vec = X[0]
    contexto_vecs = X[1:]
    similitudes = cosine_similarity(query_vec, contexto_vecs)
    umbral = 0.2
    frases_relevantes = [frases[i] for i in range(len(frases)) if similitudes[0][i] > umbral]
    return frases_relevantes

def priorizar_por_frecuencia(frases):
    frecuencia = Counter(frases)
    frases_ordenadas = sorted(frases, key=lambda x: frecuencia[x], reverse=True)
    return frases_ordenadas[:500]

def retrieve_documents(statement, k=500):
    frases_recuperadas, fuentes = buscar_en_wikipedia(statement)
    if not frases_recuperadas:
        return ["No se encontró información relevante en Wikipedia."], []

    frases_filtradas = filtrar_contenido(frases_recuperadas, statement)
    frases_priorizadas = priorizar_por_frecuencia(frases_filtradas)

    if not frases_priorizadas:
        # Si el filtrado no devuelve nada, usa la página con más contenido
        if frases_recuperadas:
            frases_priorizadas = frases_recuperadas[:k]  # Limita al número máximo permitido

    return frases_priorizadas, fuentes[:len(frases_priorizadas)]

def evaluate_truthfulness(statement):
    retrieved_docs, fuentes = retrieve_documents(statement)
    contexto = "\n".join([f"- {frase} (Fuente: {fuente})" for frase, fuente in zip(retrieved_docs, fuentes)])
    print(contexto)
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                'role': 'user',
                'content': f"Contexto:\n{contexto}\n\nPregunta: ¿Es verdadera la siguiente declaración?.\nDeclaración: {statement}\nExplica por qué basándote solo en el contexto proporcionado. Si no hay suficiente información, intenta estimar con base en el contexto, pero justifica tu respuesta:"
            },
        ]
    )

    generated_response = response['message']['content']

    if "no tengo suficiente información" in generated_response.lower() or "no puedo responder" in generated_response.lower():
        is_truthful = False
    else:
        is_truthful = "verdadero" in generated_response.lower() or "true" in generated_response.lower() or "verdadera" in generated_response.lower()

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
