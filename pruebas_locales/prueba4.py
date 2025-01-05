"""Basada en la prueba 3. Esta prueba se centra en indicar claramente si es verdadera o falsa una declaración, y en añadir un porcentaje de seguridad a la respuesta."""
import os
os.environ['OLLAMA_HOST'] = 'http://localhost:11434'
import wikipedia
import wikipediaapi
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import ollama
import re
from itertools import islice

wikipedia.set_lang("es")
# Inicializar Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)','es')

# Función para extraer frases significativas (n-gramas)
def extraer_ngrams(statement, n=2):
    palabras = re.findall(r'\b\w+\b', statement.lower())
    stop_words = {'el', 'la', 'los', 'las', 'de', 'y', 'en', 'a', 'que', 'un', 'una', 'con', 'por', 'para', 'es', 'al', 'del', 'se', 'su', 'lo'}
    palabras_filtradas = [palabra for palabra in palabras if palabra not in stop_words]
    
    # Generar n-gramas
    ngrams = zip(*(islice(palabras_filtradas, i, None) for i in range(n)))
    frases = [" ".join(ngram) for ngram in ngrams]
    return frases

# Función para dividir el texto en frases
def dividir_en_frases(texto):
    frases = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)\s(?=\w)', texto)
    return [frase.strip() for frase in frases if frase.strip()]

# Función mejorada para buscar artículos y extraer contenido con fuentes
def buscar_en_wikipedia(statement):
    frases = extraer_ngrams(statement, n=2)  # Puedes ajustar a tri-gramas con n=3
    busqueda = " ".join(frases)

    titulos = wikipedia.search(busqueda, results=15)  # Limitar a las 5 páginas más relevantes
    if not titulos:
        return None, []

    textos_recuperados = []
    fuentes = []

    for titulo in titulos:
        page = wiki_wiki.page(titulo)
        if page.exists():
            texto = page.text[:2000]  # Limitar cada página a los primeros 2000 caracteres
            frases = dividir_en_frases(texto)
            textos_recuperados.extend(frases)
            fuentes.extend([titulo] * len(frases))  # Asociar cada frase a su fuente
    return textos_recuperados, fuentes

# Función para recuperar las frases más importantes junto con sus fuentes
def retrieve_documents(statement, k=50):
    frases_recuperadas, fuentes = buscar_en_wikipedia(statement)
    if not frases_recuperadas:
        return ["No se encontró información relevante en Wikipedia."], []

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(frases_recuperadas)

    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X.toarray())

    query_vec = vectorizer.transform([statement]).toarray()
    D, I = index.search(query_vec, k)

    frases_relevantes = [frases_recuperadas[i] for i in I[0]]
    fuentes_relevantes = [fuentes[i] for i in I[0]]
    return frases_relevantes, fuentes_relevantes

# Validar si una respuesta está respaldada por el contexto de manera permisiva
def validar_respuesta(contexto, respuesta):
    texto_contexto = " ".join(contexto).lower()
    oraciones_respuesta = dividir_en_frases(respuesta.lower())

    # Considerar respaldada si al menos una parte significativa de la respuesta está en el contexto
    respaldada = any(oracion in texto_contexto for oracion in oraciones_respuesta)
    return respaldada

# Evaluación de veracidad con formato claro para el modelo
def evaluate_truthfulness(statement):
    retrieved_docs, fuentes = retrieve_documents(statement)
    contexto = "\n".join([f"- {frase} (Fuente: {fuente})" for frase, fuente in zip(retrieved_docs, fuentes)])
    print(contexto)
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                'role': 'user',
                'content': f"Contexto:\n{contexto}\n\nPregunta: ¿Es verdadera o falsa la siguiente declaración?\nDeclaración: {statement}\nResponde únicamente con 'Verdadero' o 'Falso', seguido de un porcentaje de seguridad, y explica brevemente basándote en el contexto proporcionado. Ejemplo: 'Verdadero (85% seguridad): [Explicación]'."
            },
        ]
    )

    generated_response = response['message']['content']

    respaldada = validar_respuesta(retrieved_docs, generated_response)

    if not respaldada:
        generated_response += "\nNota: La respuesta generada no está completamente respaldada por el contexto proporcionado."  

    is_truthful = "verdadero" in generated_response.lower()

    return {
        "statement": statement,
        "response": generated_response,
        "is_truthful": is_truthful,
        "sources": fuentes,
        "backed_by_context": respaldada
    }

# Ejemplo de uso
statement = input("Por favor, ingresa una afirmación para verificar su veracidad: ")

result = evaluate_truthfulness(statement)

# Imprimir resultados
print(f"Declaración: {result['statement']}")
print(f"Justificación: {result['response']}")
print(f"Fuentes utilizadas: {', '.join(set(result['sources']))}")
print(f"¿Respaldada por el contexto?: {'Sí' if result['backed_by_context'] else 'No'}")

