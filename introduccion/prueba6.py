import os
os.environ['OLLAMA_HOST'] = 'http://kumo01.tsc.uc3m.es:11434'
import ollama
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import requests
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from translate import Translator

# Paso 1: Recuperación de información

# Leer el contenido del documento
with open('Reino visigodo.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

# Dividir el contenido en párrafos
paragraphs = corpus.split('\n')

# Crear un vectorizador TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(paragraphs)
client = ollama.Client()


# Crear una base de datos vectorial usando FAISS
index = faiss.IndexFlatL2(X.shape[1])
index.add(X.toarray())

def retrieve_documents(query, k=5):
    query_vec = vectorizer.transform([query]).toarray()
    D, I = index.search(query_vec, k)
    return [paragraphs[i] for i in I[0]]

#Paso 1.2 Traducir
@Language.factory("language_detector")
def create_language_detector(nlp, name):
    return LanguageDetector(language_detection_function=None)

#python -m spacy download xx_sent_ud_sm
mult_nlp = spacy.load('xx_sent_ud_sm')
mult_nlp.add_pipe('language_detector', last=True)

def traducir_frase(frase,destino='es'):
    mult_doc = mult_nlp(frase)
    idioma = mult_doc._.language['language']
    if idioma != destino:
        translator= Translator(from_lang= idioma,to_lang=destino)
        translated = translator.translate(mult_doc.text)
        return translated,idioma
    else:
        return frase,idioma

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
                'content': f"Contexto: {context}\n\nPregunta: ¿Es verdadera la siguiente declaración?\nDeclaración: {statement}\nExplica por qué basándote solo en el contexto proporcionado. Si no hay suficiente información, indica que no puedes responder:"
            },
        ]
    )
    
    # Extraer la respuesta generada
    generated_response = response['message']['content']
    
    # Determinar si el modelo considera verdadera la declaración basándose en la respuesta
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
#statement = "El reino visigodo de Tolosa comenzó en el año 417."
statement = input("Por favor, ingresa una afirmación para verificar su veracidad: ")

# Mirar si hay que traducir
frase,idioma = traducir_frase(statement)
print(frase)

#Generar la respuesta
result = evaluate_truthfulness(frase)
statement_tranlated,_ = traducir_frase(result['statement'],idioma)
# Imprimir resultados
print(f"Declaración: {statement_tranlated}")

response_tranlated =''
i=500
while(len(result['response'])>i):
    response,_ = traducir_frase(result['response'][i-500:i],idioma)
    response_tranlated += response
    response_tranlated += " "
    i +=500
response_tranlated += traducir_frase(result['response'][i-500:i],idioma)

# print(f"Respuesta: {result['is_truthful']}")
print(f"Justificación: {response_tranlated}")

