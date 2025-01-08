import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Stopwords
import nltk

nltk.download("stopwords", quiet=True)

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
from wikipedia_utils import *
import templates


@Language.factory("language_detector")
def create_language_detector(nlp, name):
    return LanguageDetector(language_detection_function=None)


mult_nlp = spacy.load("xx_sent_ud_sm")
mult_nlp.add_pipe("language_detector", last=True)


def traducir_frase(frase, destino="en"):
    mult_doc = mult_nlp(frase)
    idioma = mult_doc._.language["language"]
    if idioma != destino:
        translator = Translator(from_lang=idioma, to_lang=destino)
        translated = translator.translate(mult_doc.text)
        return translated, idioma
    else:
        return frase, idioma


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


def setup_chain(template, input_variables, output_key):
    """
    Configura una cadena de LLM con un template, variables de entrada y clave de salida.

    Args:
        template (str): La plantilla para la cadena de LLM.
        input_variables (list): Las variables de entrada para la plantilla.
        output_key (str): La clave de salida para extraer el resultado.

    Returns:
        LLMChain: Una cadena de LLM configurada con el template, variables de entrada y clave de salida.
    """
    prompt_template = PromptTemplate(input_variables=input_variables, template=template)
    chain = LLMChain(
        llm=llm, prompt=prompt_template, output_key=output_key, verbose=verbose
    )
    return chain


def main():
    # Step 1: Question chain
    print("Please enter a statement or question:")
    statement = input("> ")
    question, idioma = traducir_frase(statement)
    idioma = iso639.to_name(idioma)

    pages, _ = buscar_en_wikipedia(question)
    sections_and_page = [("", "")]
    if pages is None:
        print("No information on the given statement.")
    else:
        sections_and_page = [
            (page.title, extraer_titulos_sections(page))
            for page in pages
            if page.exists()
        ]

    # Step 1: Detect sections
    relevant_sections_json = llm_structured._call(
        prompt=templates.get_relevant_sections.format(question, sections_and_page),
        format=section_format,
    )
    relevant_sections_dict = json.loads(relevant_sections_json)
    context = (
        extract_text_from_sections(relevant_sections_dict.get("pages", []))
        if relevant_sections_dict.get("pages")
        else {}
    )

    # Step 2: Make question based on sections
    question_chain = setup_chain(
        template=templates.questions,
        input_variables=["question", "context"],
        output_key="statement",
    )

    # Step 2: Assumptions chain with RAG
    assumptions_chain = setup_chain(
        template=templates.assumptions,
        input_variables=["statement", "context"],
        output_key="assertions",
    )

    # Step 3: Fact checker chain
    fact_checker_chain = setup_chain(
        template=templates.fact_checker,
        input_variables=["assertions"],
        output_key="facts",
    )

    # Step 4: Answer chain based on verified facts
    answer_chain = setup_chain(
        template="{facts}\n\n" + templates.answer.format(question, idioma),
        input_variables=["facts", "question"],
        output_key="final_answer",
    )

    # Step 5: Combine all the chains into a sequential workflow
    single_input_chain = SequentialChain(
        chains=[question_chain, assumptions_chain, fact_checker_chain, answer_chain],
        input_variables=["question", "context"],
        output_variables=["final_answer"],
        verbose=verbose,
    )

    # Run the entire workflow
    inputs = {"question": question, "context": context}
    intermediate_response = single_input_chain.invoke(inputs)

    # Final response
    final_response = answer_chain.run(question)
    print("FINAL RESPONSE:\n", final_response)
    print("\nDOCUMENTATION:\n Wikipedia articles used:")
    if not relevant_sections_dict.get("pages"):
        print("\tNo relevant pages found.")
    else:
        for page in relevant_sections_dict["pages"]:
            print(f"\t* {page['page_title']} § {page['section']}")


if __name__ == "__main__":
    main()
