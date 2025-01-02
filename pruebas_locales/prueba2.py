"""Esta prueba añade un control más exahustivo de la veracidad de las frases"""
import os
os.environ['OLLAMA_HOST'] = 'http://localhost:11434'
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import re
import ollama
import wikipedia
import wikipediaapi

wikipedia.set_lang("es")
# Inicializar Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)','es')


# Función para dividir el texto en frases
def dividir_en_frases(texto):
    frases = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', texto)
    return [frase.strip() for frase in frases if frase.strip()]

# Función mejorada para buscar artículos y extraer contenido con fuentes
def buscar_en_wikipedia(statement):
    titulos = wikipedia.search(statement, results=25)
    textos_recuperados = []
    fuentes = []

    for titulo in titulos:
        print(titulo)
        page = wiki_wiki.page(titulo)
        if page.exists():
            frases = dividir_en_frases(page.text[:5000])
            textos_recuperados.extend(frases)
            fuentes.extend([titulo] * len(frases))  # Asociar cada frase a su fuente
    return textos_recuperados, fuentes

# Función para recuperar las frases más importantes junto con sus fuentes
def retrieve_documents(statement, k=25):
    frases_recuperadas, fuentes = buscar_en_wikipedia(statement)
    if not frases_recuperadas:
        return ["No se encontró información relevante en Wikipedia."], []

    # Crear un vectorizador TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(frases_recuperadas)

    # Crear un índice FAISS
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X.toarray())

    # Consultar las frases más relevantes
    query_vec = vectorizer.transform([statement]).toarray()
    D, I = index.search(query_vec, k)

    # Retornar las frases relevantes con sus fuentes
    frases_relevantes = [frases_recuperadas[i] for i in I[0]]
    fuentes_relevantes = [fuentes[i] for i in I[0]]
    return frases_relevantes, fuentes_relevantes

# Evaluación de veracidad con inclusión de fuentes
def evaluate_truthfulness(statement):
    retrieved_docs, fuentes = retrieve_documents(statement)
    # Formatear contexto con frases y fuentes
    contexto = "\n".join([f"- {frase} (Fuente: {fuente})" for frase, fuente in zip(retrieved_docs, fuentes)])

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                'role': 'user',
                'content': f"Contexto:\n{contexto}\n\nPregunta: ¿Es verdadera la siguiente declaración?\nDeclaración: {statement}\nExplica por qué basándote solo en el contexto proporcionado. Si no hay suficiente información, indica que no puedes responder:"
            },
        ]
    )

    generated_response = response['message']['content']

    # Determinar si es verdadera o no
    if "no tengo suficiente información" in generated_response.lower() or "no puedo responder" in generated_response.lower():
        is_truthful = False
    else:
        is_truthful = "verdadero" in generated_response.lower() or "true" in generated_response.lower()

    return {
        "statement": statement,
        "response": generated_response,
        "is_truthful": is_truthful,
        "sources": fuentes
    }

# Ejemplo de uso
statement = input("Por favor, ingresa una afirmación para verificar su veracidad: ")

result = evaluate_truthfulness(statement)

# Imprimir resultados
print(f"Declaración: {result['statement']}")
print(f"Justificación: {result['response']}")
print(f"Fuentes utilizadas: {', '.join(set(result['sources']))}")