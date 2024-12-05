import os
os.environ['OLLAMA_HOST'] = 'http://kumo01.tsc.uc3m.es:11434'
import ollama
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

# Paso 1: Recuperación de información

# Leer el contenido del documento
with open('Reino visigodo.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

# Dividir el contenido en párrafos
paragraphs = corpus.split('\n')

# Crear un vectorizador TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(paragraphs)

# Crear una base de datos vectorial usando FAISS
index = faiss.IndexFlatL2(X.shape[1])
index.add(X.toarray())

def retrieve_documents(query, k=5):
    query_vec = vectorizer.transform([query]).toarray()
    D, I = index.search(query_vec, k)
    return [paragraphs[i] for i in I[0]]

# Paso 2: Evaluación de veracidad

def evaluate_truthfulness(statement):
    # Recuperar documentos relevantes
    retrieved_docs = retrieve_documents(statement)
    context = " ".join(retrieved_docs)
    
    # Consultar el modelo con contexto y la declaración
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                'role': 'user',
                'content': f"Contexto: {context}\n\nPregunta: ¿Es verdadera la siguiente declaración?\nDeclaración: {statement}\nExplica por qué:"
            },
        ]
    )
    
    # Extraer la respuesta generada
    generated_response = response['message']['content']
    
    # Determinar si el modelo considera verdadera la declaración basándose en la respuesta
    is_truthful = "verdadero" in generated_response.lower() or "true" in generated_response.lower()
    
    return {
        "statement": statement,
        "response": generated_response,
        "is_truthful": is_truthful
    }

# Ejemplo de uso
#statement = "El reino visigodo de Tolosa comenzó en el año 417."
statement = input("Por favor, ingresa una afirmación para verificar su veracidad: ")

result = evaluate_truthfulness(statement)

# Imprimir resultados
print(f"Declaración: {result['statement']}")
#print(f"Respuesta: {result['is_truthful']}")
print(f"Justificaciñon: {result['response']}")

