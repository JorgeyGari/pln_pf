import os
os.environ['OLLAMA_HOST'] = 'http://kumo01.tsc.uc3m.es:11434'
import ollama
import wikipediaapi
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from transformers import pipeline
import numpy as np
import wikipedia


# --- BASE DE DATOS ---
def search_wikipedia(query, lang='es'):
    wikipedia.set_lang(lang)
    wikipedia.set_user_agent("MyBot/1.0 (https://mywebsite.com/; mybot@example.com) wikipedia-library/1.0")
    try:
        page = wikipedia.page(query)
        return page.content
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Se encontraron varias opciones para '{query}': {e.options}"
    except wikipedia.exceptions.PageError:
        return "Artículo no encontrado."
    except Exception as e:
        return f"Error inesperado: {str(e)}"

# --- FUNCIONALIDADES ADICIONALES ---
def calculate_confidence_score(contexts, statement):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(contexts + [statement])
    cosine_similarities = (X[-1] * X[:-1].T).toarray().flatten()
    return np.mean(cosine_similarities)

def generate_summary(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(text, max_length=130, min_length=30, do_sample=False)

# --- FUNCIONES PARA PROCESAR EL CONTEXTO ---
def retrieve_context(statement, corpus, k=5):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([statement]).toarray()
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X.toarray())
    D, I = index.search(query_vec, k)
    return [corpus[i] for i in I[0] if i < len(corpus)]

def process_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = file.readlines()
    return [line.strip() for line in corpus if line.strip()]

# --- EVALUAR LA VERACIDAD ---
def evaluate_truthfulness(statement, combined_context):
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                'role': 'user',
                'content': f"""
                Usa exclusivamente el siguiente contexto para responder:
                Contexto: {combined_context}
                Pregunta: Según el contexto proporcionado, ¿es verdadera o falsa la siguiente declaración?
                Declaración: "{statement}"
                Responde exclusivamente con "Verdadero" o "Falso", seguido de una breve justificación.
                """
            }
        ]
    )
    generated_response = response['message']['content']
    is_truthful = "verdadero" in generated_response.lower() or "true" in generated_response.lower()
    return {
        "statement": statement,
        "response": generated_response,
        "is_truthful": is_truthful
    }

# --- PROCESO PRINCIPAL ---
def main():
    # Cargar archivo de preguntas
    question_file = "pregunta.txt"
    with open(question_file, 'r', encoding='utf-8') as file:
        question = file.readline().strip()

    # Procesar corpus local
    text_file = "Reino visigodo.txt"
    corpus = process_text(text_file)

    # Recuperar contexto de Wikipedia
    print("Seleccione fuente de datos:\n1. Corpus local\n2. Wikipedia")
    source = int(input("Opción: "))
    if source == 1:
        combined_context = " ".join(retrieve_context(question, corpus, k=5))
    elif source == 2:
        combined_context = search_wikipedia(question)
    else:
        print("Opción no válida. Usando corpus local por defecto.")
        combined_context = " ".join(retrieve_context(question, corpus, k=5))

    # Evaluar veracidad
    result = evaluate_truthfulness(question, combined_context)

    # Mostrar resultados
    print("\nDeclaración:", result["statement"])
    print("Respuesta generada:", result["response"])
    print("Respuesta:", result["is_truthful"])

    # Calcular confianza y resumen
    confidence = calculate_confidence_score([combined_context], question)
    summary = generate_summary(combined_context[:1024])  #No tiene mucho sentido lo del resumen, pero creo que lo pide el enunciado
    print("\nPuntaje de confianza:", confidence)
    print("\nResumen:", summary[0]['summary_text'])

if __name__ == "__main__":
    main()
