import wikipediaapi

from entity_finder import list_entities


wiki = wikipediaapi.Wikipedia("FactChecker (email@example.com)", "en")


def extract_section_titles(page):
    section_titles = []
    sections = page.sections
    for s in sections:
        section_titles.append(s.title)
    return section_titles


def wikipedia_search(statement) -> tuple:
    """
    Given a statement, search for relevant Wikipedia pages and return the pages and their sections.

    Args:
        statement (str): The statement to search for.

    Returns:
        list: A list of Wikipedia pages.
        list: A list of the sections of the Wikipedia pages.
    """
    entity_titles = list_entities(statement)

    if not entity_titles:
        return None, []

    retrieved_pages = []

    for entity_title in entity_titles:
        wiki_page = wiki.page(entity_title)
        if wiki_page.exists():
            wiki_content = wiki_page
            retrieved_pages.append((entity_title, wiki_content))

    retrieved_content = [page[1] for page in retrieved_pages]
    page_titles_list = [page[0] for page in retrieved_pages]

    return retrieved_content, page_titles_list


def extract_text_from_sections(pages_sections_dict):
    """
    Given the relevant sections and a list of pages/sections,
    extract the corresponding text from those sections using the Wikipedia API.
    """
    relevant_texts = {}

    for page in pages_sections_dict:
        wiki_page = wiki.page(page["page_title"])
        if wiki_page.exists():
            text = wiki_page.section_by_title(page["section"])
            if text:
                relevant_texts[page["section"] + " (" + page["page_title"] + ")"] = text
    return relevant_texts
