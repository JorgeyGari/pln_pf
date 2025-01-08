import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
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
import iso639

# Custom imports
from config import *
from entity_finder import list_entities
from wikipedia_utils import wiki_wiki

@Language.factory("language_detector")
def create_language_detector(nlp, name):
    return LanguageDetector(language_detection_function=None)

mult_nlp = spacy.load('xx_sent_ud_sm')
mult_nlp.add_pipe('language_detector', last=True)

def traducir_frase(frase,destino='en'):
    mult_doc = mult_nlp(frase)
    idioma = mult_doc._.language['language']
    if idioma != destino:
        translator= Translator(from_lang= idioma,to_lang=destino)
        translated = translator.translate(mult_doc.text)
        return translated,idioma
    else:
        return frase,idioma

def extraer_titulos_sections(page):
    array_titulos = []
    sections = page.sections
    for s in sections:
        array_titulos.append(s.title)
    return array_titulos

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

def main():
    # Step 1: Question chain
    statement = "¿Francisco de Goya pintó 'Las Meninas'?"
    question, idioma = traducir_frase(statement)
    print(f"Se ha traducido la frase {statement}\n({idioma}>en) {question}")
    idioma = iso639.to_name(idioma)

    pages, sources = buscar_en_wikipedia(question)
    sections_and_page = [
        (page.title, extraer_titulos_sections(page)) 
        for page in pages 
        if page is not None and page.exists()
    ]

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
                        "section": {
                            "type": "string",
                        }
                    },
                    "required": ["page_title", "section"]
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
    Determine the pages most relevant to answer the question. Specify the sections that are most relevant to answer the question. Keep the number of sections to a minimum. You may repeat pages if necessary to retrieve multiple sections from the same page.
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
            text = wiki_wiki.page(page["page_title"]).section_by_title(page["section"])
            if text:
                relevant_texts[page["section"] + " (" + page["page_title"] + ")"] = text
        return relevant_texts


    context = extract_text_from_sections(response_dict["pages"])

    # Step 3: Make question based on sections
    template_q = """Given the following question: {question}\n Answer it based on the following information: {context}\n\n"""
    prompt_template_q = PromptTemplate(input_variables=["question", "context"], template=template_q)
    question_chain = LLMChain(llm=llm, prompt=prompt_template_q, output_key="statement", verbose=False)

    # Step 2: Assumptions chain with RAG
    template_assumptions = """Here is a statement:
    {statement}
    Make a bullet point list of the assumptions you made when producing the above statement based on the following documents:
    {context}\n\n"""
    prompt_template_assumptions = PromptTemplate(input_variables=["statement", "context"], template=template_assumptions)
    assumptions_chain = LLMChain(llm=llm, prompt=prompt_template_assumptions, output_key="assertions", verbose=False)

    # Step 3: Fact checker chain
    template_fact_checker = """Here is the bullet point list of assertions: \n
    {assertions}\n\n
    For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
    prompt_template_fact_checker = PromptTemplate(input_variables=["assertions"], template=template_fact_checker)
    fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template_fact_checker, output_key="facts", verbose=False)

    # Step 4: Answer chain based on verified facts
    template_answer = f"""In light of the above facts, how would you answer the question '{question}'"""
    template_answer = """{facts}\n""" + template_answer + f"\nPlease respond in {idioma}."
    prompt_template_answer = PromptTemplate(input_variables=["facts"], template=template_answer)
    answer_chain = LLMChain(llm=llm, prompt=prompt_template_answer, output_key = "final_answer", verbose=False)

    # Step 5: Combine all the chains into a sequential workflow
    # Combine chains with single input
    single_input_chain = SequentialChain(
        chains=[question_chain, assumptions_chain, fact_checker_chain, answer_chain],
        input_variables=["question", "context"],
        output_variables=["final_answer"],
        verbose=False
    )


    # Run the entire workflow
    inputs = {"question": question, "context": context}
    intermediate_response = single_input_chain.run(inputs)

    # Final response
    final_response = answer_chain.run(question)
    print("FINAL RESPONSE:\n", final_response)
    sources = list(dict.fromkeys(sources))
    print("DOCUMENTATION:\n-", "\n-".join(sources))

if __name__ == "__main__":
    main()