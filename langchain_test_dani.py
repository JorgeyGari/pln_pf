from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, TransformChain, SequentialChain
from typing import Optional
from http import HTTPStatus
import wikipediaapi
import wikipedia
import re
from itertools import islice
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
# STOPWORDS
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
#TRANSLATION
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from translate import Translator
import ast

# Paso 1.1 Inicializar Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')

#Paso 1.2 Traducir
@Language.factory("language_detector")
def create_language_detector(nlp, name):
    return LanguageDetector(language_detection_function=None)

#python -m spacy download xx_sent_ud_sm
mult_nlp = spacy.load('xx_sent_ud_sm')
mult_nlp.add_pipe('language_detector', last=True)

# Diccionario de abreviaturas de idiomas a nombres completos en inglés
language_map = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}

def get_language_name(abbreviation):
    return language_map.get(abbreviation, "English")

def traducir_frase(frase,destino='en'):
    mult_doc = mult_nlp(frase)
    idioma = mult_doc._.language['language']
    if idioma != destino:
        translator= Translator(from_lang= idioma,to_lang=destino)
        translated = translator.translate(mult_doc.text)
        return translated,idioma
    else:
        return frase,idioma

# Paso 1.3. Función para extraer frases significativas (n-gramas)
def extraer_ngrams(statement, n=2):
    palabras = re.findall(r'\b\w+\b', statement.lower())
    stop_words = set(stopwords.words('english'))
    palabras_filtradas = [palabra for palabra in palabras if palabra not in stop_words]
    
    # Generar n-gramas
    ngrams = zip(*(islice(palabras_filtradas, i, None) for i in range(n)))
    frases = [" ".join(ngram) for ngram in ngrams]
    return frases

# Paso 1.4. Función para dividir el texto en frases
def dividir_en_frases(texto):
    frases = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)\s(?=\w)', texto)
    return [frase.strip() for frase in frases if frase.strip()]

def print_sections(sections, level=0):
        for s in sections:
                print("%s: %s - %s" % ("*" * (level + 1), s.title, s.text))
                print_sections(s.sections, level + 1)

def extraer_titulos_sections(page):
    array_titulos = []
    sections = page.sections
    for s in sections:
        array_titulos.append(s.title)
    return array_titulos

    

# Paso 1.5. Función mejorada para buscar artículos y extraer contenido con fuentes
def buscar_en_wikipedia(statement):
    frases = extraer_ngrams(statement, n=2)
    busqueda = " ".join(frases)

    titulos = wikipedia.search(busqueda, results=3)

    if not titulos:
        return None, []

    paginas_contenido = []

    for titulo in titulos:
        page = wiki_wiki.page(titulo)
        if page.exists():
            pagina = page
            paginas_contenido.append((titulo, pagina))


    paginas_recuperadas = [pagina[1] for pagina in paginas_contenido]
    fuentes = [pagina[0] for pagina in paginas_contenido]

    return paginas_recuperadas, fuentes

# Paso 1.6. Función para recuperar las frases más importantes junto con sus fuentes
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

def string_to_dict(input_string):
    """
    Convierte una cadena con formato específico en un diccionario.
    
    Args:
        input_string (str): La cadena con el formato específico.

    Returns:
        dict: Un diccionario con las claves y listas correspondientes.
    """
    # Convierte la cadena en una tupla utilizando ast.literal_eval para evaluar la estructura segura
    try:
        parsed_tuple = ast.literal_eval(input_string)
    except (ValueError, SyntaxError):
        raise ValueError("El formato de la cadena no es válido.")

    result_dict = {}

    # Recorre la tupla para organizarla en el diccionario
    for item in parsed_tuple:
        if isinstance(item, str):
            # Crea claves con valores vacíos inicialmente
            result_dict[item] = []
        elif isinstance(item, list):
            # Asocia la última clave con el contenido de la lista
            last_key = list(result_dict.keys())[-1]
            result_dict[last_key] = item
        else:
            raise ValueError("El formato de los elementos de la tupla no es válido.")

    return result_dict

def extract_text_from_sections(info):
    extracted_info = string_to_dict(info['info_output'])
    section_texts = []
    for page_title, sections in extracted_info.items():
        for page in pages:
            if page.title == page_title:
                for section in page.sections:
                    if section.title in sections:
                        section_texts.append(section.text)
    return {"context": section_texts}

# Paso 2 Clase central
class CustomLLM(LLM):
    model: str
    system: str
    api_url: str = "http://localhost:11434/api/generate"

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
    model="llama3.2:latest",
    system="You are a helpful AI Assistant",
)

# Step 1: Question chain
# statement = "Is Mercury the closest planet to the sun."
statement = "¿Francisco de Goya pintó 'Las Meninas'?"
# Mirar si hay que traducir
question,idioma = traducir_frase(statement)
print(f"Se ha traducido la frase {statement}\n({idioma}-en)-> {question}")
idioma = get_language_name(idioma)


# Execute the entire workflow with document retrieval
pages, sources = buscar_en_wikipedia(question)
sections_and_page = [(page.title, extraer_titulos_sections(page)) for page in pages]
print(sections_and_page)
print(f"Question: {question}\n")


# Step 1: Detect sections
"""Detecta cuál es la sección más importante para la pregunta"""
template_get_info = """Identify the most relevant sections from a set of pages to answer the given question.

Inputs:
1. **Question:** "{question}"
2. **Pages and Sections:** {sections_and_page}

Task:
Determine the sections that are most relevant to answer the question and respond strictly in the following format:

Output Format:
[name_of_page1, [section1, section2, ...], name_of_page2, [section1, ...], ...]

Do not include any explanations or additional text.

"""
prompt_template_info = PromptTemplate(input_variables=["question", "sections_and_page"], template=template_get_info)
info_chain = LLMChain(llm=llm, prompt=prompt_template_info, output_key="info_output", verbose=True)



#Step 2: Extract text from sections
transform_chain = TransformChain(
    input_variables=["info_output"],
    output_variables=["context"],
    transform=extract_text_from_sections,
    verbose=True,
)

# Step 3: Make question based on sections
template_q = """Given the following question: {question}\n Answer it based on the following context: {context}\n\n"""
prompt_template_q = PromptTemplate(input_variables=["question", "context"], template=template_q)
question_chain = LLMChain(llm=llm, prompt=prompt_template_q, output_key="statement", verbose=True)

# Step 2: Assumptions chain with RAG
template_assumptions = """Here is a statement:
{statement}
Make a bullet point list of the assumptions you made when producing the above statement based on the following documents:
{context}\n\n"""
prompt_template_assumptions = PromptTemplate(input_variables=["statement", "context"], template=template_assumptions)
assumptions_chain = LLMChain(llm=llm, prompt=prompt_template_assumptions, output_key="assertions", verbose=True)

# Step 3: Fact checker chain
template_fact_checker = """Here is the bullet point list of assertions:
{assertions}
For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
prompt_template_fact_checker = PromptTemplate(input_variables=["assertions"], template=template_fact_checker)
fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template_fact_checker, output_key="facts", verbose=True)

# Step 4: Answer chain based on verified facts
template_answer = """In light of the above facts, how would you answer the question '{}'""".format(question)
template_answer = """{facts}\n""" + template_answer + f"\nPlease respond in {idioma}."
prompt_template_answer = PromptTemplate(input_variables=["facts"], template=template_answer)
answer_chain = LLMChain(llm=llm, prompt=prompt_template_answer, output_key = "final_answer", verbose=True)

# Step 5: Combine all the chains into a sequential workflow
# Combine chains with single input
single_input_chain = SequentialChain(
    chains=[info_chain, transform_chain, question_chain, assumptions_chain, fact_checker_chain, answer_chain],
    input_variables=["question", "sections_and_page"],
    output_variables=["final_answer"],
    verbose=True
)


inputs = {"question": question, "sections_and_page": sections_and_page}
# Run the single input chain first
intermediate_response = single_input_chain.run(inputs)

# Run the assumptions chain separately
final_response = answer_chain.run(question)
print("FINAL RESPONSE:\n",final_response)
sources = list(dict.fromkeys(sources))
print("DOCUMENTATION:\n-","\n-".join(sources))