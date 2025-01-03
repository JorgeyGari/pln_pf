from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from typing import Optional
from http import HTTPStatus
import wikipediaapi
import wikipedia
import re
from itertools import islice
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Inicializar Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')

# Función para extraer frases significativas (n-gramas)
def extraer_ngrams(statement, n=2):
    palabras = re.findall(r'\b\w+\b', statement.lower())
    stop_words = set(stopwords.words('english'))
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

    titulos = wikipedia.search(busqueda, results=15)  # Limitar a las 15 páginas más relevantes
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

class CustomLLM(LLM):
    model: str
    system: str
    api_url: str = "http://kumo01.tsc.uc3m.es:11434/api/generate"

    def _call(self, prompt: str, stop: Optional[list] = None) -> str:
        import requests
        import json

        data = {
            "model": self.model,
            "system": self.system,
            "prompt": prompt,
            "stream": False,
        }
        if stop:
            data["stop"] = stop
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
            if response.status_code == HTTPStatus.OK:
                result = response.json()
                return result.get("response", "")
            else:
                raise ValueError(f"Error: {response.status_code}, {response.text}")
        except Exception as e:
            raise ValueError(f"An error occurred: {str(e)}")

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

    @property
    def _identifying_params(self):
        return {"model": self.model, "system": self.system}

llm = CustomLLM(
    model="llama3.1:latest",
    system="You are a helpful AI Assistant",
)

# Step 1: Question chain
question = "The Roman Empire collapsed in the 5th Century."
template_q = """{question}\n\n"""
prompt_template_q = PromptTemplate(input_variables=["question"], template=template_q)
question_chain = LLMChain(llm=llm, prompt=prompt_template_q)

# Step 2: Assumptions chain with RAG
template_assumptions = """Here is a statement:
{statement}
Make a bullet point list of the assumptions you made when producing the above statement based on the following documents:
{documents}\n\n"""
prompt_template_assumptions = PromptTemplate(input_variables=["statement", "documents"], template=template_assumptions)
assumptions_chain = LLMChain(llm=llm, prompt=prompt_template_assumptions)

# Step 3: Fact checker chain
template_fact_checker = """Here is the bullet point list of assertions:
{assertions}
For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
prompt_template_fact_checker = PromptTemplate(input_variables=["assertions"], template=template_fact_checker)
fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template_fact_checker)

# Step 4: Answer chain based on verified facts
template_answer = """In light of the above facts, how would you answer the question '{}'""".format(question)
template_answer = """{facts}\n""" + template_answer
prompt_template_answer = PromptTemplate(input_variables=["facts"], template=template_answer)
answer_chain = LLMChain(llm=llm, prompt=prompt_template_answer)

# Step 5: Combine all the chains into a sequential workflow
# Combine chains with single input
single_input_chain = SimpleSequentialChain(
    chains=[question_chain, fact_checker_chain, answer_chain],
    verbose=True
)

# Execute the entire workflow with document retrieval
documents, sources = retrieve_documents(question)
print(f"Question: {question}\n")

# Run the single input chain first
intermediate_response = single_input_chain.run(question)

# Run the assumptions chain separately
final_response = answer_chain.run(question)
print("FINAL RESPONSE:\n",final_response)
sources = list(dict.fromkeys(sources))
print("DOCUMENTATION:\n-","\n-".join(sources))