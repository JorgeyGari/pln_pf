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
    retrieved_docs = retrieve_documents(statement)
    context = " ".join(retrieved_docs)
    
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                'role': 'user',
                'content': f"Contexto: {context}\n\nPregunta: ¿Es verdadera la siguiente declaración?\nDeclaración: {statement}\nRespuesta:",
            },
        ]
    )
    
    generated_response = response['message']['content']
    
    # Verificar si la respuesta generada está contenida en los documentos recuperados
    is_truthful = statement in context
    
    return {
        "statement": statement,
        "response": generated_response,
        "is_truthful": is_truthful
    }

# Ejemplo de uso del sistema de verificación de hechos
statement = "El reino visigodo de Toledo comenzó en el año 507."
result = evaluate_truthfulness(statement)

print(f"Declaración: {result['statement']}")
print(f"Respuesta: {result['response']}")
print(f"Es verdadera: {result['is_truthful']}")
