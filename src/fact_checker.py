import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, TransformChain, SequentialChain
import wikipediaapi
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

# Stopwords
import nltk
nltk.download('stopwords', quiet=True)

# Translation
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from translate import Translator
import ast

# Custom imports
from config import *
from entity_finder import list_entities
from pydantic_schema import validate_info_output

# Validation and Extraction
def extract_text_from_sections(section_dict):
    # Extract relevant section texts
    extracted_info = string_to_dict(section_dict['info_output'])
    section_texts = []
    for page_title, sections in extracted_info.items():
        for page in pages:
            if page.title == page_title:
                for section in page.sections:
                    if section.title in sections:
                        section_texts.append(section.text)
    
    print(f"Extracted section texts: {section_texts}")
    return {"context": section_texts}

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
    titulos = list_entities(statement)

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
    _, I = index.search(query_vec, k)

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



# Step 1: Question chain
# statement = "Is the Sun the closest planet to the Sun?"
statement = "¿Francisco de Goya pintó 'Las Meninas'?"
# Mirar si hay que traducir
question, idioma = traducir_frase(statement)
print(f"Se ha traducido la frase {statement}\n({idioma}-en)-> {question}")
idioma = get_language_name(idioma)


# Execute the entire workflow with document retrieval
pages, sources = buscar_en_wikipedia(question)
sections_and_page = [
    (page.title, extraer_titulos_sections(page)) 
    for page in pages 
    if page is not None and page.exists()
]
print("Sections and page:\n", sections_and_page)
print(f"Question: {question}\n")


# Step 1: Detect sections
llm_structured = CustomLLM(
    model="llama3.2:latest",
    system="You are a helpful AI Assistant",
)
format = {
    "type": "object",
    "properties": {
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page_title": {
                        "type": "string"
                    },
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["page_title", "sections"]
            }
        }
    },
    "required": ["pages"]
}
response = llm_structured._call(prompt=f"""
Identify the most relevant sections from a set of pages to answer the given question.

The second input given is a list of tuples of pages and sections in the following format:
[(name_of_page1, [section1, section2, ...]), (name_of_page2, [section1, ...]), ...]

Inputs:
1. **Question:** "{question}"
2. **Pages and Sections:** {sections_and_page}

Task:
Determine the pages most relevant to answer the question. Specify the sections that are most relevant to answer the question. Try to keep the number of sections to a minimum.
""", format=format)

# Parse response JSON into a dictionary
response_dict = json.loads(response)
print("Relevant sections:\n", response_dict)

def extract_text_from_sections(pages_sections_dict):
    """
    Given the relevant sections and a list of pages/sections,
    extract the corresponding text from those sections using the Wikipedia API.
    """
    relevant_texts = {}
    
    for page in pages_sections_dict:
        print("Page:", page)
        for section in page["sections"]:
            print("Section:", section)
            # Extract text from the section
            text = wiki_wiki.page(page["page_title"]).section_by_title(section)
            if text:
                relevant_texts[section + " (" + page["page_title"] + ")"] = text
            else:
                print(f"Section '{section}' not found in page '{page['page_title']}'")
    return relevant_texts


context = extract_text_from_sections(response_dict["pages"])

print("Context:\n", context)

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
    chains=[question_chain, assumptions_chain, fact_checker_chain, answer_chain],
    input_variables=["question", "context"],
    output_variables=["final_answer"],
    verbose=True
)


# Run the entire workflow
inputs = {"question": question, "context": context}
intermediate_response = single_input_chain.run(inputs)

# Final response
final_response = answer_chain.run(question)
print("FINAL RESPONSE:\n", final_response)
sources = list(dict.fromkeys(sources))
print("DOCUMENTATION:\n-", "\n-".join(sources))