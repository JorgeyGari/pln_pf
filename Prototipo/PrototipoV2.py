import os
os.environ['OLLAMA_HOST'] = 'http://kumo01.tsc.uc3m.es:11434'
import ollama
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from bert_score import score

# Configuración inicial


# Función para procesar el texto fuente
def process_text(text_file):
    with open(text_file, 'r', encoding='utf-8') as file:
        corpus = file.read()
    paragraphs = [line.split(" ", 1)[1] if line[0].isdigit() else line for line in corpus.split("\n") if line.strip()]
    return paragraphs

# Recuperar documentos relevantes
def retrieve_documents(query, paragraphs, k=5):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(paragraphs)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X.toarray())
    query_vec = vectorizer.transform([query]).toarray()
    _, D = index.search(query_vec, k)
    return [paragraphs[i] for i in D[0] if i < len(paragraphs)]

# Evaluación de veracidad
def evaluate_truthfulness(statement, context):
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                'role': 'user',
                'content': f"""
                Según el contexto proporcionado, responde si la declaración es "Verdadero" o "Falso":
                Declaración: "{statement}"

                - Si la declaración es parcialmente correcta, considera que es "Verdadero".
                - Justificación: Proporciona un único párrafo justificando tu respuesta.
                """
            }
        ]
    )

    # Extraer respuesta generada
    generated_response = response['message']['content']
    is_truthful = "verdadero" in generated_response.lower() or "true" in generated_response.lower()

    return {
        "statement": statement,
        "response": generated_response,
        "is_truthful": is_truthful
    }

# Métricas de evaluación
def calculate_metrics(generated_response, reference_response):
    P, R, F1 = score([generated_response], [reference_response], lang="es")
    metrics = {
        "Precision": P.item(),
        "Recall": R.item(),
        "F1": F1.item()
    }
    return metrics

# Leer pregunta de archivo
def read_question_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readline().strip()

# Función principal
def main():
    text_file = "Reino visigodo.txt"
    question_file = "pregunta.txt"
    
    # Procesar texto fuente
    paragraphs = process_text(text_file)

    # Preguntar al usuario
    user_input = input("Ingrese una pregunta o escriba '0' para cargarla desde el archivo: ").strip()
    
    if user_input == '0':
        statement = read_question_from_file(question_file)
        print(f"Pregunta cargada desde archivo: {statement}")
    else:
        statement = user_input

    # Recuperar contexto relevante
    context = " ".join(retrieve_documents(statement, paragraphs, k=5))
    print(f"Contexto recuperado:\n{context}")

    # Evaluar veracidad
    result = evaluate_truthfulness(statement, context)
    
    # Imprimir resultados
    print(f"\nDeclaración: {result['statement']}")
    print(f"Respuesta generada: {result['response']}")
    print(f"Es verdadera: {'Verdadero' if result['is_truthful'] else 'Falso'}")

    # Calcular métricas
    reference_response = context  # Usar el contexto como referencia para evaluación
    metrics = calculate_metrics(result['response'], reference_response)
    print("\nMétricas de evaluación:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# Ejecutar
if __name__ == "__main__":
    main()
