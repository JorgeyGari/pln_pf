"""Esta prueba comprueba el funcionamiento del modelo dentro de un entorno local"""
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

# Función para buscar artículos en Wikipedia usando términos destacados
def buscar_en_wikipedia(statement):
    # Extraer frases relevantes (bi-gramas o tri-gramas)
    frases = extraer_ngrams(statement, n=2)  # Puedes ajustar a tri-gramas con n=3
    busqueda = " ".join(frases[:3])  # Usar las primeras 3 frases para la búsqueda

    # Buscar títulos en Wikipedia
    titulos = wikipedia.search(busqueda, results=15)
    if not titulos:
        return None, []

    # Recuperar contenido de los títulos encontrados
    textos_recuperados = []
    for titulo in titulos:
        print(titulo)
        page = wiki_wiki.page(titulo)
        if page.exists():
            textos_recuperados.append(page.text[:10000])  # Limitar a 5000 caracteres
    return textos_recuperados

# Función para recuperar los documentos más relevantes usando TF-IDF y FAISS
def retrieve_documents(statement, k=50):
    textos_recuperados = buscar_en_wikipedia(statement)
    if not textos_recuperados:
        return ["No se encontró información relevante en Wikipedia."]
    
    paragraphs = []
    for texto in textos_recuperados:
        paragraphs.extend(texto.split('\n'))

    # Crear un vectorizador TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(paragraphs)

    # Crear una base de datos vectorial usando FAISS
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X.toarray())

    # Consultar los párrafos más relevantes
    query_vec = vectorizer.transform([statement]).toarray()
    D, I = index.search(query_vec, k)

    return [paragraphs[i] for i in I[0]]

# Paso 2: Evaluación de veracidad
def evaluate_truthfulness(statement):
    retrieved_docs = retrieve_documents(statement)
    context = " ".join(retrieved_docs)
    print("Contexto recuperado:")
    print(context)
    print("-----------------------------------")

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                'role': 'user',
                'content': f"Contexto: {context}\n\nPregunta: ¿Es verdadera la siguiente declaración?\nDeclaración: {statement}\nExplica por qué basándote solo en el contexto proporcionado. Es posible que no toda la información del contexto sea útil. Indica las fuentes de donde sacas la información. Si no hay suficiente información, indica que no puedes responder:"
            },
        ]
    )

    generated_response = response['message']['content']

    if "no tengo suficiente información" in generated_response.lower() or "no puedo responder" in generated_response.lower():
        is_truthful = False
    else:
        is_truthful = "verdadero" in generated_response.lower() or "true" in generated_response.lower()

    return {
        "statement": statement,
        "response": generated_response,
        "is_truthful": is_truthful
    }

# Ejemplo de uso
statement = input("Por favor, ingresa una afirmación para verificar su veracidad: ")

result = evaluate_truthfulness(statement)

# Imprimir resultados
print(f"Declaración: {result['statement']}")
print(f"Justificación: {result['response']}")