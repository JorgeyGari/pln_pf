import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import warnings

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

warnings.filterwarnings(
    "default" if ALLOW_WARNINGS else "ignore"
)  # Suppress LangChain deprecation warnings


@Language.factory("language_detector")
def create_language_detector(nlp, name):
    return LanguageDetector(language_detection_function=None)


mult_nlp = spacy.load("xx_sent_ud_sm")
mult_nlp.add_pipe("language_detector", last=True)


def translate_input(input_sentence, target_language="en") -> tuple:
    """
    Translates the input sentence to the target language if it is not already in the target language.
    """
    mult_doc = mult_nlp(input_sentence)
    original_language = mult_doc._.language["language"]
    if original_language != target_language:
        translator = Translator(from_lang=original_language, to_lang=target_language)
        translated = translator.translate(mult_doc.text)
        return translated, original_language
    else:
        return input_sentence, original_language


def string_to_dict(input_string) -> dict:
    """
    Converts a properly formatted string to a dictionary.
    """
    try:
        parsed_tuple = ast.literal_eval(input_string)
    except (ValueError, SyntaxError):
        raise ValueError("Invalid input format.")

    result_dict = {}

    for item in parsed_tuple:
        if isinstance(item, str):
            result_dict[item] = []
        elif isinstance(item, list):
            last_key = list(result_dict.keys())[-1]
            result_dict[last_key] = item
        else:
            raise ValueError("Tuple format is invalid.")

    return result_dict


def setup_chain(template, input_variables, output_key) -> LLMChain:
    """
    Setup a chain with the given template and input variables.

    Args:
        template (str): The template to use.
        input_variables (list): The input variables to use.
        output_key (str): The output key to use.

    Returns:
        LLMChain: The chain with the specified template and input variables.
    """
    prompt_template = PromptTemplate(input_variables=input_variables, template=template)
    chain = LLMChain(
        llm=llm, prompt=prompt_template, output_key=output_key, verbose=VERBOSE
    )
    return chain


def main():
    # Step 1: Process input
    print("Please enter a statement or question:")
    statement = input("> ")
    question, lang = translate_input(statement)
    lang = iso639.to_name(lang)

    pages, _ = wikipedia_search(question)
    sections_and_page = [("", "")]
    if pages is None:
        print("No information on the given statement.")
    else:
        sections_and_page = [
            (page.title, extract_section_titles(page))
            for page in pages
            if page.exists()
        ]

    # Step 2: Detect relevant sections and retrieve text from Wikipedia
    relevant_sections_json = llm._call(
        prompt=templates.get_relevant_sections.format(question, sections_and_page),
        format=section_format,
    )
    relevant_sections_dict = json.loads(relevant_sections_json)
    context = (
        extract_text_from_sections(relevant_sections_dict.get("pages", []))
        if relevant_sections_dict.get("pages")
        else {}
    )

    # Step 3: Make question based on sections
    question_chain = setup_chain(
        template=templates.questions,
        input_variables=["question", "context"],
        output_key="reasoning",
    )

    # Step 4: Answer the question
    answer_chain = setup_chain(
        template="{reasoning}\n\n" + templates.answer.format(question, lang),
        input_variables=["reasoning", "question"],
        output_key="final_answer",
    )

    # Combine all chains
    single_input_chain = SequentialChain(
        chains=[question_chain, answer_chain],
        input_variables=["question", "context"],
        output_variables=["final_answer"],
        verbose=VERBOSE,
    )

    # Run the entire workflow
    inputs = {"question": question, "context": context}
    final_answer = single_input_chain.invoke(inputs)["final_answer"]
    print("FINAL RESPONSE:\n", final_answer)
    print("\nDOCUMENTATION:\n Wikipedia articles used:")
    if not relevant_sections_dict.get("pages"):
        print("\tNo relevant pages found.")
    else:
        for page in relevant_sections_dict["pages"]:
            print(f"\t* {page['page_title']} § {page['section']}")

    # Step 5: Summarize sections
    if input("Would you like to generate a summary of the information used? (y/n): ").lower() == "y":
        summaries = llm._call(
            prompt=templates.summarize.format(context=context, question=question, language=lang),
        )
        print(summaries)
    

if __name__ == "__main__":
    main()
