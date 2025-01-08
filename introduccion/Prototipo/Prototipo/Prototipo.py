import os
os.environ['OLLAMA_HOST'] = 'http://kumo01.tsc.uc3m.es:11434'
import ollama
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss


# Leer el contenido del documento
with open('Reino visigodo.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

# Dividir el contenido en párrafos
paragraphs = corpus.split('\n\n')

# Crear un vectorizador TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(paragraphs)

# Crear una base de datos vectorial usando FAISS
index = faiss.IndexFlatL2(X.shape[1])
index.add(X.toarray())

def retrieve_documents(query, k=1):
    """Recuperar los K documentos más relevantes."""
    query_vec = vectorizer.transform([query]).toarray()
    D, I = index.search(query_vec, k)
    return [paragraphs[i] for i in I[0]]

def evaluate_truthfulness(statement):
    """Evalúa la veracidad de una declaración usando el modelo y contexto."""
    # Recuperar documentos relevantes
    retrieved_docs = retrieve_documents(statement)
    context = " ".join(retrieved_docs)
    print(f"Contexto recuperado:\n{context}\n")

    # Consultar el modelo con contexto y la declaración
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
            'role': 'user',
            'content': f"""Usa exclusivamente el siguiente contexto para responder:

Contexto: {context}

Pregunta: Según el contexto proporcionado, ¿es verdadera o falsa la siguiente declaración? 
Declaración: "{statement}"

Responde estrictamente siguiendo este formato:
1. En la primera línea, escribe únicamente una de estas opciones: "Verdadero" o "Falso".
   - Si la declaración es parcialmente correcta o tiene elementos ambiguos, clasifícala como "Verdadero".
   - Sé flexible con la evaluación. Prioriza clasificar como "Verdadero" si la esencia general de la declaración coincide con el contexto, aunque haya detalles menores incorrectos.

2. A continuación, proporciona un único párrafo justificando tu clasificación. En este párrafo:
   - Explica los aspectos clave del contexto que respaldan tu decisión.
   - Si hay diferencias menores, menciona por qué no afectan significativamente a la validez de la declaración.

No incluyas más de un párrafo en tu respuesta y no seas muy redundante."""
            },
        ]
    )

    # Extraer la respuesta generada
    generated_response = response['message']['content']
    # Determinar si el modelo considera la declaración verdadera o falsa
    is_truthful = "verdadero" in generated_response.lower() or "true" in generated_response.lower()

    return {
        "statement": statement,
        "response": generated_response,
        "is_truthful": is_truthful,
        "retrieved_docs": retrieved_docs,
        "context": context
    }

if __name__ == "__main__":
    print("Ingrese una pregunta o escriba '0' para cargarla desde el archivo:")
    user_input = input("> ")

    # Si el usuario ingresa 0, leer la pregunta desde el archivo pregunta.txt
    if user_input == "0":
        try:
            with open("pregunta.txt", "r", encoding="utf-8") as question_file:
                statement = question_file.read().strip()
                print(f"Pregunta cargada desde archivo: {statement}")
        except FileNotFoundError:
            print("Error: El archivo 'pregunta.txt' no existe. Por favor, crea el archivo con una pregunta.")
            exit(1)
    else:
        # Usar la entrada del usuario como la pregunta
        statement = user_input

    # Evaluar la veracidad
    result = evaluate_truthfulness(statement)
    print(f"\nDeclaración: {result['statement']}")
    print(f"\nRespuesta: {result['is_truthful']}")
    print(f"\nJustificación: {result['response']}")
    
   